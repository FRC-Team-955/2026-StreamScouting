import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    BOUNCE_OUT_RISE,
    FRAME_SKIP,
    END_SECONDS,
    MAX_TRAIL,
    PARABOLA_A_MIN,
    PARABOLA_MIN_POINTS,
    PARABOLA_R2_MIN,
    SCORE_COOLDOWN_FRAMES,
    SCORE_MIN_DESCENT,
    SCORE_MIN_INSIDE_POINTS,
    SKIP_SECONDS,
    TRAIL_DECAY,
)
from tracker import Tracker
from vision import (
    blackout_hole,
    blackout_outside_active,
    check_parabola_score,
    crop_frame,
    detect_circles,
    fit_parabola,
    get_runtime_regions,
)


def run(
    video_path,
    side,
    frame_skip=FRAME_SKIP,
    start_seconds=None,
    score_min_descent=SCORE_MIN_DESCENT,
    score_min_inside_points=SCORE_MIN_INSIDE_POINTS,
    bounce_out_rise=BOUNCE_OUT_RISE,
    parabola_r2_min=PARABOLA_R2_MIN,
    headless=False,
):
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Could not read input video dimensions")

    crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP_SECONDS * fps)
    end_frames = None

    if start_seconds is not None:
        start_frames = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)
        if END_SECONDS is not None:
            end_frames = int(END_SECONDS * fps)

    print(
        f"[regions] src={frame_w}x{frame_h} crop={crop_region} "
        f"hole={hole_region} score={score_polygon} active={active_region}"
    )

    tracker  = Tracker()
    trails   = defaultdict(list)  # smoothed position history per track
    t_alphas = defaultdict(list)  # opacity per trail point, decays over time
    first_seen = {}               # first position ever seen for each track, before trail decay can erase it
    origins    = {}               # launch point per track, only shown once confirmed scored

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -10)
    scored_track_ids        = set()

    fps_window   = 30
    frame_times  = []
    display_fps  = 0.0
    last_fps_update = time.time()

    frame_idx = int(start_seconds * fps) if start_seconds is not None else 0

    while True:
        # grab() seeks without decoding, much faster than read() for skipped frames
        for _ in range(max(0, frame_skip - 1)):
            cap.grab()
            frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if end_frames is not None and frame_idx >= end_frames:
            break

        if frame_idx < skip_frames:
            continue

        t_start = time.perf_counter()

        frame   = crop_frame(frame, crop_region)
        circles = detect_circles(frame, hole_region, active_region)
        objects = tracker.update([(x, y) for (x, y, r) in circles])

        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                if oid not in first_seen:
                    first_seen[oid] = (x, y)
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # multiply all alphas by decay factor each frame, drop points that have faded out
        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps = zip(*combined)
                trails[oid]   = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)

        for oid, pts in trails.items():
            track      = tracker.tracks.get(oid)
            track_lost = bool(track and track.ghost_count == 1)
            if check_parabola_score(
                    oid,
                    pts,
                    frame_idx,
                    last_score_frame_per_id,
                    last_score_frame,
                    score_polygon,
                    scored_track_ids,
                    track_lost=track_lost,
                    score_min_descent=score_min_descent,
                    score_min_inside_points=score_min_inside_points,
                    bounce_out_rise=bounce_out_rise,
            ):
                score += 1
                last_score_frame_per_id[oid] = frame_idx
                last_score_frame             = frame_idx
                scored_track_ids.add(oid)
                # save the launch point now that we know it was a scored shot
                if oid not in origins and oid in first_seen:
                    origins[oid] = first_seen[oid]
                print(f"  [SCORE]  ID {oid} @ frame {frame_idx} -> total: {score}")

        vis = frame.copy()
        vis = blackout_outside_active(vis, active_region)
        vis = blackout_hole(vis, hole_region)

        cv2.polylines(vis, [np.array(score_polygon, dtype=np.int32)], True, (0, 255, 0), 2)

        for (x, y, r) in circles:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        for oid in trails:
            pts  = trails[oid]
            alps = t_alphas[oid]
            for i in range(1, len(pts)):
                a     = alps[i - 1]
                color = (0, int(255 * a), int(255 * (1 - a)))
                cv2.line(vis, pts[i - 1], pts[i], color, 2)

        for oid, pts in trails.items():
            if len(pts) >= PARABOLA_MIN_POINTS:
                result = fit_parabola(
                    pts,
                    parabola_min_points=PARABOLA_MIN_POINTS,
                    parabola_a_min=PARABOLA_A_MIN,
                )
                if result:
                    a, b, c, r2 = result
                    xs     = np.array([p[0] for p in pts])
                    x_mean = xs.mean()
                    col    = (0, 200, 255) if r2 > parabola_r2_min else (80, 80, 80)
                    for xi in range(int(xs.min()), int(xs.max()), 2):
                        xn  = xi - x_mean
                        yi  = int(a * xn**2 + b * xn + c)
                        xn2 = xi + 2 - x_mean
                        yi2 = int(a * xn2**2 + b * xn2 + c)
                        h, w = vis.shape[:2]
                        if 0 <= yi < h and 0 <= yi2 < h:
                            cv2.line(vis, (xi, yi), (xi + 2, yi2), col, 1)

        for tid, track in tracker.tracks.items():
            if track.ghost_count > 0:
                px, py = track.position()
                alpha  = 1.0 - track.ghost_count / 4
                color  = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(vis, (px, py), 6, color, 1)

        for oid, (x, y) in objects.items():
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(oid), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # draw launch point markers for scored shots, persists for the rest of the video
        for oid, (ox, oy) in origins.items():
            arm, gap = 8, 3
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cv2.line(
                    vis,
                    (ox + dx * gap,         oy + dy * gap),
                    (ox + dx * (arm + gap), oy + dy * (arm + gap)),
                    (0, 200, 255), 2, cv2.LINE_AA,
                )
            cv2.circle(vis, (ox, oy), 3, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, f"#{oid}", (ox + arm + gap + 2, oy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        now = time.time()
        if now - last_fps_update > 0.5 and frame_times:
            display_fps     = 1.0 / (sum(frame_times) / len(frame_times))
            last_fps_update = now

        cv2.putText(vis, f"Score: {score}", (460, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"{display_fps:.1f}/{(60 / max(1, frame_skip)):.0f} fps  tracks: {len(tracker.tracks)}",
                    (460, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        if not headless:
            cv2.imshow("tracking", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track")
    parser.add_argument("--side", type=str, choices=["red", "blue"], default="red")
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--video-file", type=str)
    parser.add_argument("--start", type=float)
    parser.add_argument("--score-min-descent", type=int, default=SCORE_MIN_DESCENT)
    parser.add_argument("--score-min-inside-points", type=int, default=SCORE_MIN_INSIDE_POINTS)
    parser.add_argument("--bounce-out-rise", type=int, default=BOUNCE_OUT_RISE)
    parser.add_argument("--parabola-r2-min", type=float, default=PARABOLA_R2_MIN)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    frame_skip = args.frame_drop if args.frame_drop is not None else FRAME_SKIP

    print("Initializing...")
    sc = run(
        args.video_file,
        args.side,
        frame_skip=frame_skip,
        start_seconds=args.start,
        score_min_descent=args.score_min_descent,
        score_min_inside_points=args.score_min_inside_points,
        bounce_out_rise=args.bounce_out_rise,
        parabola_r2_min=args.parabola_r2_min,
        headless=args.headless,
    )
    print(f"Final score: {sc}")