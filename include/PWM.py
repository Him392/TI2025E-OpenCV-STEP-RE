from typing import Dict, Optional


AXIS_MAP = {
    3: "pan",
    4: "tilt",
}


class ServoController:
    """
    Servo target output facade.

    This class no longer drives PWM directly. It only records and exposes the
    latest target angle for each channel so a later serial stepper backend can
    forward those targets to external hardware.
    """

    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username or ""
        self.password = password or ""
        self.target_angles: Dict[str, int] = {}
        self.last_command: Optional[dict] = None
        print(
            "ServoController initialized in target-only mode. "
            "No PWM output will be generated."
        )

    def check_user_permissions(self):
        return True

    def ensure_pwm_permissions(self, servonum: int):
        return True

    def _axis_name(self, servonum: int) -> str:
        return AXIS_MAP.get(servonum, f"servo_{servonum}")

    def servoset(self, servonum: int, angle: int):
        clamped_angle = max(0, min(1023, int(angle)))
        axis = self._axis_name(servonum)
        self.target_angles[axis] = clamped_angle
        self.last_command = {"axis": axis, "target_angle": clamped_angle}
        print(f"[ServoController] {self.last_command}")
        return dict(self.last_command)

    def get_target_angle(self, servonum: int, default: Optional[int] = None):
        axis = self._axis_name(servonum)
        return self.target_angles.get(axis, default)

    def get_all_target_angles(self):
        return dict(self.target_angles)

    def get_last_command(self):
        return dict(self.last_command) if self.last_command is not None else None

    def servo_release(self, servonum: int):
        axis = self._axis_name(servonum)
        if axis in self.target_angles:
            del self.target_angles[axis]
            self.last_command = {"axis": axis, "target_angle": None}
            print(f"[ServoController] released {self.last_command}")
        else:
            print(f"[ServoController] axis={axis} was not active")


if __name__ == "__main__":
    controller = ServoController()
    controller.servoset(servonum=3, angle=480)
    controller.servoset(servonum=4, angle=512)
    print(controller.get_all_target_angles())
    controller.servo_release(servonum=3)
    controller.servo_release(servonum=4)
