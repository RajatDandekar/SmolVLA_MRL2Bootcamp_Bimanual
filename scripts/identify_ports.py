"""
Identify which USB port corresponds to which arm.
Connects to both ports and continuously reads motor positions.
Move one arm by hand to see which port responds.

Usage:
    source activate.sh
    python scripts/identify_ports.py
"""

import time
import sys

def main():
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    ports = [
        "/dev/tty.wchusbserial5AE60829961",
        "/dev/tty.wchusbserial5AE60840931",
    ]

    norm = MotorNormMode.RANGE_M100_100
    motors = {
        "shoulder_pan": Motor(1, "sts3215", norm),
        "shoulder_lift": Motor(2, "sts3215", norm),
        "elbow_flex": Motor(3, "sts3215", norm),
        "wrist_flex": Motor(4, "sts3215", norm),
        "wrist_roll": Motor(5, "sts3215", norm),
        "gripper": Motor(6, "sts3215", norm),
    }

    buses = []
    for port in ports:
        try:
            bus = FeetechMotorsBus(port=port, motors=motors)
            bus.connect()
            buses.append((port, bus))
            print(f"Connected to {port}")
        except Exception as e:
            print(f"Failed to connect to {port}: {e}")

    if len(buses) < 2:
        print("Could not connect to both ports. Check connections.")
        for _, bus in buses:
            bus.disconnect()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Move ONE arm by hand. Watch which port shows changing values.")
    print("Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        while True:
            for port, bus in buses:
                positions = bus.sync_read("Present_Position")
                vals = [f"{v:5d}" for v in positions.tolist()]
                short_port = port.split("serial")[-1]
                print(f"  {short_port}: {vals}")
            print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for _, bus in buses:
            bus.disconnect()
        print("Disconnected all ports.")

if __name__ == "__main__":
    main()
