#!/usr/bin/env python3
"""Click points on a video frame to get pixel coordinates for zones / stop line / calibration.

Usage: python tools/pick_points.py path/to/video.mp4 [frame_number]
Left-click prints the coordinate as [x, y]; press 'q' to quit.
"""
import sys

import cv2


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cap = cv2.VideoCapture(sys.argv[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read a frame")

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("[%d, %d]" % (x, y))
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.namedWindow("pick points (q to quit)")
    cv2.setMouseCallback("pick points (q to quit)", on_click)
    while True:
        cv2.imshow("pick points (q to quit)", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
