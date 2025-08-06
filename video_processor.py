import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import rfdetr

from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker


# ────────────────────────────── helpers ────────────────────────────────────── #
def open_source(src: str) -> cv2.VideoCapture:
    """Accept webcam index or path."""
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def is_display_available() -> bool:
    """Check if a display is available for GUI operations."""
    # Check for common headless indicators
    if os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
        return False
    
    # Platform-specific checks
    if sys.platform == 'linux':
        return bool(os.environ.get('DISPLAY'))
    elif sys.platform == 'win32':
        return True  # Windows typically has display
    elif sys.platform == 'darwin':
        return True  # macOS typically has display
    
    return True  # Default to assuming display exists


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSORT++ single-camera demo")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--out", default=None, help="optional output video path")
    p.add_argument("--log-interval", type=float, default=1.0,
                   help="seconds between log lines")
    p.add_argument("--headless", action="store_true",
                   help="force headless mode (no GUI)")
    p.add_argument("--gui", action="store_true",
                   help="force GUI mode (override auto-detection)")
    p.add_argument("--max-frames", type=int, default=500,
                   help="maximum frames to process (useful for headless)")
    p.add_argument("--save-interval", type=int, default=None,
                   help="save frame images every N frames (headless mode)")
    p.add_argument("--save-dir", default="frames",
                   help="directory for saved frames")
    return p.parse_args()


class VideoProcessor:
    """Encapsulates video processing logic with headless/GUI flexibility."""
    
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.log = logger
        self.use_gui = self._determine_gui_mode()
        
        # Initialize components
        self.cap = None
        self.writer = None
        self.tracker = None
        self.frame_saver = None
        
        if not self.use_gui:
            self.log.info("Running in HEADLESS mode")
            if args.save_interval:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                self.log.info(f"Saving frames every {args.save_interval} to {args.save_dir}/")
        else:
            self.log.info("Running in GUI mode")
    
    def _determine_gui_mode(self) -> bool:
        """Determine whether to use GUI based on args and environment."""
        if self.args.headless:
            return False
        if self.args.gui:
            return True
        return is_display_available()
    
    def initialize(self) -> bool:
        """Initialize video capture, writer, and tracker."""
        self.cap = open_source(self.args.source)
        if not self.cap.isOpened():
            self.log.error("Could not open source %s", self.args.source)
            return False
        
        w, h = int(self.cap.get(3)), int(self.cap.get(4))
        fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        self.log.info(f"Video source: {w}x{h} @ {fps_src:.1f} FPS")
        
        # Output writer
        if self.args.out:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(Path(self.args.out)), fourcc, fps_src, (w, h))
            self.log.info(f"Writing output to {self.args.out}")
        
        # Tracker
        self.tracker = SingleCameraTracker(
            od_name="yolo",
            tracker_kwargs=dict(max_age=50, iou_thres=0.4, appearance_thres=0.5),
        )
        
        return True
    
    def process_frame(self, frame, frame_cnt: int) -> tuple:
        """Process a single frame and return tracks."""
        tracks = self.tracker.update(frame)
        self.tracker.draw(frame, tracks)
        
        # Save frame periodically in headless mode
        if not self.use_gui and self.args.save_interval:
            if frame_cnt % self.args.save_interval == 0:
                fname = Path(self.args.save_dir) / f"frame_{frame_cnt:06d}.jpg"
                cv2.imwrite(str(fname), frame)
                self.log.debug(f"Saved {fname}")
        
        return frame, tracks
    
    def display_frame(self, frame) -> bool:
        """Display frame if GUI available. Returns False if should exit."""
        if not self.use_gui:
            return True  # Continue processing
        
        cv2.imshow("DeepSORT", frame)
        return cv2.waitKey(1) & 0xFF != 27  # ESC to exit
    
    def run(self):
        """Main processing loop."""
        if not self.initialize():
            return
        
        frame_cnt = 0
        start = last_log = time.perf_counter()
        
        def _persistent(track):
            return track[5] >= 2  # hits index = 5
        
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    self.log.info("End of video stream")
                    break
                
                # Process
                frame, tracks = self.process_frame(frame, frame_cnt)
                
                # Write output
                if self.writer:
                    self.writer.write(frame)
                
                # Display and check for exit
                if not self.display_frame(frame):
                    self.log.info("User requested exit")
                    break
                
                # Stats logging
                frame_cnt += 1
                now = time.perf_counter()
                if now - last_log >= self.args.log_interval:
                    fps_cur = frame_cnt / (now - start)
                    persistent = sum(_persistent(t) for t in tracks)
                    self.log.info(
                        f"Frame {frame_cnt} | FPS: {fps_cur:.1f} | "
                        f"Objects: {persistent} (persistent≥2 hits)"
                    )
                    last_log = now
                
                # Frame limit for headless mode
                if self.args.max_frames and frame_cnt >= self.args.max_frames:
                    self.log.info(f"Reached max frames ({self.args.max_frames})")
                    break
                
        except KeyboardInterrupt:
            self.log.info("Interrupted by user")
        except Exception as e:
            self.log.error(f"Processing error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.use_gui:
            cv2.destroyAllWindows()
        self.log.info("Cleanup complete")


# ────────────────────────────── main ──────────────────────────────────────── #
def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    log = logging.getLogger("single-cam")
    
    # Run processor
    processor = VideoProcessor(args, log)
    processor.run()


if __name__ == "__main__":
    main()