#!/usr/bin/env python3
"""
Advanced Real-Time YOLO Object Detector
========================================
Usage examples:
  # Webcam with default settings
  python main.py

  # Video file with YOLO11s, only detect person & car
  python main.py --source data/sample.mp4 --model yolo11s.pt --classes person car

  # Webcam with segmentation task
  python main.py --model yolo11n-seg.pt --task segment

  # Define a counting line (draws a virtual line; objects crossing it are counted)
  python main.py --line 0,360 1280,360

  # Record output to file
  python main.py --record exports/output.mp4

  # ROI zone – only detect inside the polygon
  python main.py --roi 200,100 1000,100 1000,600 200,600

Keyboard shortcuts (while window is open):
  Q / ESC  – quit
  S        – save screenshot
  R        – toggle recording on/off
  P        – pause / resume
  +        – raise confidence threshold by 0.05
  -        – lower confidence threshold by 0.05
  H        – toggle help overlay
  F        – toggle trails
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

from src.detector import AdvancedDetector
from src.utils    import SmoothFPS, VideoRecorder, SnapshotSaver, draw_hud, resize_frame


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Advanced Real-Time YOLO Object Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # source
    p.add_argument("--source", default="0",
        help="Video source: '0' for webcam, path to video/image file, or RTSP URL")

    # model
    p.add_argument("--model", default="models/yolo11n.pt",
        help="YOLO11 model path or tag (e.g. yolo11n.pt, yolo11s.pt, yolo11n-seg.pt)")
    p.add_argument("--task", default="detect",
        choices=["detect", "segment", "pose", "obb"],
        help="Model task (default: detect)")

    # thresholds
    p.add_argument("--conf", type=float, default=0.40,
        help="Confidence threshold [0-1]")
    p.add_argument("--iou", type=float, default=0.50,
        help="NMS IoU threshold [0-1]")

    # tracker
    p.add_argument("--tracker", default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm")

    # device
    p.add_argument("--device", default=None,
        help="Inference device: cuda | mps | cpu (default: auto)")

    # filters
    p.add_argument("--classes", nargs="*", default=None,
        help="Class names to detect (e.g. --classes person car)")

    # ROI
    p.add_argument("--roi", nargs="*", default=None,
        help="ROI polygon vertices as x,y pairs (e.g. 100,100 900,100 900,600 100,600)")

    # line counter
    p.add_argument("--line", nargs=2, default=None, metavar=("PT1", "PT2"),
        help="Counting line: two x,y points (e.g. --line 0,360 1280,360)")

    # display
    p.add_argument("--width", type=int, default=0,
        help="Resize display to this width (0 = native)")
    p.add_argument("--no-trails", action="store_true",
        help="Disable object trails")
    p.add_argument("--trail-length", type=int, default=30,
        help="Length of object trails in frames")

    # performance
    p.add_argument("--skip", type=int, default=0,
        help="Frame skip: run inference every N+1 frames")

    # output
    p.add_argument("--record", default=None,
        help="Output video path (recording starts immediately)")
    p.add_argument("--snapshot-dir", default="exports",
        help="Directory for screenshots")

    return p.parse_args()


def parse_point(s: str):
    x, y = s.split(",")
    return (int(x), int(y))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── video source ──────────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Source] {src_w}×{src_h} @ {src_fps:.1f} fps")

    # ── ROI / line ────────────────────────────────────────────────────────────
    roi_points  = None
    line_points = None

    if args.roi:
        roi_points = [parse_point(p) for p in args.roi]

    if args.line:
        line_points = (parse_point(args.line[0]), parse_point(args.line[1]))

    # ── detector ──────────────────────────────────────────────────────────────
    print("[Init] Loading model …")
    detector = AdvancedDetector(
        model_path      = args.model,
        conf_threshold  = args.conf,
        iou_threshold   = args.iou,
        tracker         = args.tracker,
        device          = args.device,
        target_classes  = args.classes,
        roi_points      = roi_points,
        line_points     = line_points,
        frame_skip      = args.skip,
        show_trails     = not args.no_trails,
        trail_length    = args.trail_length,
        task            = args.task,
    )
    print(f"[Init] {detector.model_info()}")

    # ── optional recorder ─────────────────────────────────────────────────────
    disp_w = args.width if args.width > 0 else src_w
    disp_h = int(src_h * disp_w / src_w) if args.width > 0 else src_h

    recorder: VideoRecorder | None = None
    recording = False

    if args.record:
        recorder = VideoRecorder(args.record, src_fps, disp_w, disp_h)
        recording = True
        print(f"[Record] Writing to {args.record}")

    # ── misc helpers ──────────────────────────────────────────────────────────
    fps_counter  = SmoothFPS(window=30)
    snapshotter  = SnapshotSaver(args.snapshot_dir)
    show_help    = True
    paused       = False

    print("\nPress H to toggle help overlay. Press Q or ESC to quit.\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Info] End of stream.")
                break

            if args.width > 0:
                frame = resize_frame(frame, args.width)

            # inference + annotation
            annotated = detector.detect(frame)

            # HUD overlay
            fps = fps_counter.update()
            annotated = draw_hud(
                annotated,
                fps        = fps,
                inf_ms     = detector.avg_inference_ms(),
                device     = detector.device,
                show_controls = show_help,
            )

            # recording
            if recording and recorder:
                recorder.update(annotated)

            cv2.imshow("Advanced YOLO Detector  [H] Help  [Q] Quit", annotated)
        else:
            # paused – just show last frame
            cv2.waitKey(50)
            continue

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):          # Q / ESC → quit
            break

        elif key == ord("s"):              # S → screenshot
            snapshotter.save(annotated)

        elif key == ord("p"):              # P → pause / resume
            paused = not paused
            print("[Paused]" if paused else "[Resumed]")

        elif key == ord("h"):              # H → toggle help
            show_help = not show_help

        elif key == ord("f"):              # F → toggle trails
            detector.show_trails = not detector.show_trails
            print(f"[Trails] {'ON' if detector.show_trails else 'OFF'}")

        elif key == ord("r"):              # R → toggle recording
            if recorder is None:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(args.snapshot_dir, f"record_{ts}.mp4")
                recorder = VideoRecorder(path, src_fps, disp_w, disp_h)
                recording = True
                print(f"[Record] Started → {path}")
            else:
                recorder.close()
                recorder  = None
                recording = False
                print("[Record] Stopped")

        elif key == ord("+") or key == ord("="):   # + → raise conf
            detector.conf_threshold = min(0.95, round(detector.conf_threshold + 0.05, 2))
            print(f"[Conf] {detector.conf_threshold:.2f}")

        elif key == ord("-"):              # - → lower conf
            detector.conf_threshold = max(0.05, round(detector.conf_threshold - 0.05, 2))
            print(f"[Conf] {detector.conf_threshold:.2f}")

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if recorder:
        recorder.close()
    cv2.destroyAllWindows()

    # final stats
    print(f"\n[Stats] Total detections   : {detector.total_detections}")
    print(f"[Stats] Avg inference time : {detector.avg_inference_ms()} ms")
    print(f"[Stats] Device used        : {detector.device}")
    if detector.line_counter:
        lc = detector.line_counter
        print(f"[Stats] Line counter       : IN={lc.in_count}  OUT={lc.out_count}")


if __name__ == "__main__":
    main()
