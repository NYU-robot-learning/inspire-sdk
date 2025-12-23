import argparse
import base64
import pickle
import signal
import sys
import time

import cv2
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to Aria stream data")
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        default="100.94.225.27",
        help="Host address of the publisher",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=10012,
        help="Port of the publisher",
    )
    parser.add_argument(
        "--subscribe-rgb",
        dest="subscribe_rgb",
        action="store_true",
        default=True,
        help="Subscribe to RGB images",
    )
    parser.add_argument(
        "--subscribe-landmarks",
        dest="subscribe_landmarks",
        action="store_true",
        default=True,
        help="Subscribe to hand landmarks",
    )
    parser.add_argument(
        "--display-rgb",
        dest="display_rgb",
        action="store_true",
        default=False,
        help="Display RGB images in a window",
    )
    parser.add_argument(
        "--print-landmarks",
        dest="print_landmarks",
        action="store_true",
        default=False,
        help="Print hand landmarks to console",
    )
    return parser.parse_args()


class AriaStreamSubscriber:
    def __init__(self, host, port, subscribe_rgb=True, subscribe_landmarks=True, 
                 display_rgb=False, print_landmarks=False):
        self.host = host
        self.port = port
        self.subscribe_rgb = subscribe_rgb
        self.subscribe_landmarks = subscribe_landmarks
        self.display_rgb = display_rgb
        self.print_landmarks = print_landmarks
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        if self.subscribe_rgb:
            self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image ")
            print("Subscribed to RGB images")
        
        if self.subscribe_landmarks:
            self.socket.setsockopt(zmq.SUBSCRIBE, b"hand_landmarks ")
            print("Subscribed to hand landmarks")
        
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to publisher at tcp://{host}:{port}")
        
        if self.display_rgb:
            cv2.namedWindow("Aria RGB Stream", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Aria RGB Stream", cv2.WND_PROP_TOPMOST, 1)
        
        self.rgb_count = 0
        self.landmarks_count = 0
        self.last_stats_time = time.time()
        
    def process_rgb_message(self, data):
        self.rgb_count += 1
        
        base64_data = data["rgb_image"]
        timestamp_ns = data["timestamp"]
        
        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if rgb is None:
            print("Failed to decode RGB image")
            return
        
        if self.display_rgb:
            cv2.imshow("Aria RGB Stream", rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        return True
    
    def process_landmarks_message(self, data):
        self.landmarks_count += 1
        
        print(f"\n=== Hand Landmarks (timestamp: {data['timestamp']:.6f} s) ===")
        
        if data["left_hand"]:
            left = data["left_hand"]
            print(f"Left hand - Confidence: {left['confidence']:.3f}")
            print(f"  Wrist translation: {left['wrist_translation']}")
            print(f"  Wrist quaternion: {left['wrist_quaternion']}")
            print(f"  Landmarks count: {len(left['landmarks_device'])}")
            if self.print_landmarks and 'landmarks_device' in left:
                print(f"  Landmarks (first 5):")
                for i, landmark in enumerate(left['landmarks_device'][:5]):
                    print(f"    [{i}] {landmark}")
        else:
            print("Left hand: Not detected")
        
        if data["right_hand"]:
            right = data["right_hand"]
            print(f"Right hand - Confidence: {right['confidence']:.3f}")
            print(f"  Wrist translation: {right['wrist_translation']}")
            print(f"  Wrist quaternion: {right['wrist_quaternion']}")
            print(f"  Landmarks count: {len(right['landmarks_device'])}")
            if self.print_landmarks and 'landmarks_device' in right:
                print(f"  Landmarks (first 5):")
                for i, landmark in enumerate(right['landmarks_device'][:5]):
                    print(f"    [{i}] {landmark}")
        else:
            print("Right hand: Not detected")
        
        return True
    
    def print_stats(self):
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed >= 5.0:
            rgb_fps = self.rgb_count / elapsed if elapsed > 0 else 0
            landmarks_fps = self.landmarks_count / elapsed if elapsed > 0 else 0
            
            print(f"\n[Stats] RGB: {self.rgb_count} frames ({rgb_fps:.2f} fps) | "
                  f"Landmarks: {self.landmarks_count} messages ({landmarks_fps:.2f} fps)")
            
            self.rgb_count = 0
            self.landmarks_count = 0
            self.last_stats_time = current_time
    
    def run(self):
        print("Starting subscriber... Press Ctrl+C to quit")
        
        try:
            while True:
                if self.socket.poll(timeout=100):
                    try:
                        # Publisher sends topic + pickled data as single concatenated message
                        raw_message = self.socket.recv(zmq.NOBLOCK)
                        
                        # Check for topic prefixes (publisher concatenates topic + pickled data)
                        if raw_message.startswith(b"rgb_image "):
                            # Extract topic and pickled data
                            topic_len = len(b"rgb_image ")
                            pickled_data = raw_message[topic_len:]
                            data = pickle.loads(pickled_data)
                            if not self.process_rgb_message(data):
                                break
                            self.print_stats()
                        elif raw_message.startswith(b"hand_landmarks "):
                            # Extract topic and pickled data
                            topic_len = len(b"hand_landmarks ")
                            pickled_data = raw_message[topic_len:]
                            data = pickle.loads(pickled_data)
                            self.process_landmarks_message(data)
                            self.print_stats()
                        else:
                            # Unknown format - try to decode anyway
                            print(f"Received message with unknown format (length: {len(raw_message)} bytes)")
                            print(f"First 50 bytes: {raw_message[:50]}")
                            try:
                                # Maybe it's just pickled data without topic?
                                data = pickle.loads(raw_message)
                                if isinstance(data, dict):
                                    if 'rgb_image' in data:
                                        if not self.process_rgb_message(data):
                                            break
                                        self.print_stats()
                                    elif 'left_hand' in data or 'right_hand' in data:
                                        self.process_landmarks_message(data)
                                        self.print_stats()
                            except Exception as e:
                                print(f"Could not decode message: {e}")
                    except zmq.Again:
                        continue
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\nStopping subscriber...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.display_rgb:
            cv2.destroyAllWindows()
        self.socket.close()
        self.context.term()
        print("Subscriber stopped")


def main():
    args = parse_args()
    
    subscriber = AriaStreamSubscriber(
        host=args.host,
        port=args.port,
        subscribe_rgb=args.subscribe_rgb,
        subscribe_landmarks=args.subscribe_landmarks,
        display_rgb=args.display_rgb,
        print_landmarks=args.print_landmarks
    )
    
    subscriber.run()


if __name__ == "__main__":
    main()
