import tkinter as tk
from tkinter import messagebox
import time
from hand_sdk import RH56RobotHand


class RobotHandGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RH56 Robot Hand Controller")
        self.root.geometry("750x650")
        self.root.resizable(False, False)
        
        # Robot hand instance
        self.hand = None
        self.connected = False
        
        # Store current values
        self.current_angles = [0, 0, 0, 0, 0, 0]
        self.current_speed = 50
        self.current_force = 50
        
        # Finger names
        self.finger_names = ["Finger 1", "Finger 2", "Finger 3", 
                            "Finger 4", "Finger 5", "Finger 6"]
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        
        # Connection Frame
        conn_frame = tk.LabelFrame(self.root, text="Connection", padx=10, pady=10)
        conn_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=5)
        self.port_entry = tk.Entry(conn_frame, width=20)
        self.port_entry.insert(0, "/dev/ttyUSB0")
        self.port_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(conn_frame, text="Baudrate:").grid(row=0, column=2, padx=5)
        self.baudrate_entry = tk.Entry(conn_frame, width=10)
        self.baudrate_entry.insert(0, "115200")
        self.baudrate_entry.grid(row=0, column=3, padx=5)
        
        self.connect_btn = tk.Button(conn_frame, text="Connect", command=self.toggle_connection,
                                     bg="#4CAF50", fg="white", width=12)
        self.connect_btn.grid(row=0, column=4, padx=10)
        
        self.status_label = tk.Label(conn_frame, text="Disconnected", fg="red")
        self.status_label.grid(row=0, column=5, padx=10)
        
        # Speed and Force Frame
        control_frame = tk.LabelFrame(self.root, text="Speed & Force Control", padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Speed slider
        tk.Label(control_frame, text="Speed:").grid(row=0, column=0, sticky="w", pady=5)
        self.speed_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     length=450, command=self.on_speed_change)
        self.speed_slider.set(50)
        self.speed_slider.grid(row=0, column=1, padx=10)
        self.speed_value_label = tk.Label(control_frame, text="50%", width=6)
        self.speed_value_label.grid(row=0, column=2)
        
        # Force slider
        tk.Label(control_frame, text="Force:").grid(row=1, column=0, sticky="w", pady=5)
        self.force_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     length=450, command=self.on_force_change)
        self.force_slider.set(50)
        self.force_slider.grid(row=1, column=1, padx=10)
        self.force_value_label = tk.Label(control_frame, text="50%", width=6)
        self.force_value_label.grid(row=1, column=2)
        
        # Finger sliders frame
        fingers_frame = tk.LabelFrame(self.root, text="Finger Control (0=Open, 100=Closed)", 
                                      padx=10, pady=10)
        fingers_frame.pack(fill="x", padx=10, pady=10)
        
        self.finger_sliders = []
        self.finger_value_labels = []
        
        for i, name in enumerate(self.finger_names):
            # Finger label
            tk.Label(fingers_frame, text=f"{name}:").grid(
                row=i, column=0, sticky="w", pady=5, padx=5)
            
            # Slider
            slider = tk.Scale(fingers_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                            length=450, command=lambda val, idx=i: self.on_finger_change(idx, val))
            slider.set(0)
            slider.grid(row=i, column=1, padx=10)
            self.finger_sliders.append(slider)
            
            # Value label
            value_label = tk.Label(fingers_frame, text="0%", width=6)
            value_label.grid(row=i, column=2)
            self.finger_value_labels.append(value_label)
        
        # Quick action buttons
        action_frame = tk.LabelFrame(self.root, text="Quick Actions", padx=10, pady=10)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        btn_frame = tk.Frame(action_frame)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Open All", command=self.open_all, width=12).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="Close All", command=self.close_all, width=12).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="Half Close", command=self.half_close, width=12).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(btn_frame, text="Reset", command=self.reset_sliders, width=12).grid(row=0, column=3, padx=5, pady=5)
        
        # Feedback frame
        feedback_frame = tk.LabelFrame(self.root, text="Sensor Feedback", padx=10, pady=10)
        feedback_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        btn_frame2 = tk.Frame(feedback_frame)
        btn_frame2.pack()
        
        tk.Button(btn_frame2, text="Read Angles", command=self.read_angles, width=12).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame2, text="Read Forces", command=self.read_forces, width=12).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame2, text="Read Temps", command=self.read_temps, width=12).grid(row=0, column=2, padx=5, pady=5)
        
        self.feedback_text = tk.Text(feedback_frame, height=5, width=80)
        self.feedback_text.pack(padx=5, pady=5)
        
    def toggle_connection(self):
        """Connect or disconnect from the robot hand"""
        if not self.connected:
            try:
                port = self.port_entry.get()
                baudrate = int(self.baudrate_entry.get())
                
                self.hand = RH56RobotHand(port=port, baudrate=baudrate, hand_id=1)
                if self.hand.connect():
                    # Set initial speed and force
                    self.hand.set_speeds([self.current_speed] * 6)
                    self.hand.set_forces([self.current_force] * 6)
                    
                    # Read actual current angles from the hand
                    angles = self.hand.get_angles()
                    if angles:
                        self.current_angles = angles
                        # Update sliders to match actual positions
                        for i, angle in enumerate(angles):
                            self.finger_sliders[i].set(angle)
                        self.log_feedback(f"Read angles: {[f'{a:.1f}%' for a in angles]}")
                    
                    self.connected = True
                    self.status_label.config(text="Connected", fg="green")
                    self.connect_btn.config(text="Disconnect", bg="#f44336")
                    self.log_feedback("Connected successfully!")
                else:
                    messagebox.showerror("Connection Error", "Failed to connect")
            except Exception as e:
                messagebox.showerror("Error", f"{str(e)}")
        else:
            if self.hand:
                self.hand.disconnect()
            self.connected = False
            self.status_label.config(text="Disconnected", fg="red")
            self.connect_btn.config(text="Connect", bg="#4CAF50")
            self.log_feedback("Disconnected")
    
    def on_speed_change(self, value):
        """Handle speed slider change"""
        speed = int(float(value))
        self.current_speed = speed
        self.speed_value_label.config(text=f"{speed}%")
        
        if self.connected and self.hand:
            self.hand.set_speeds([speed] * 6)
    
    def on_force_change(self, value):
        """Handle force slider change"""
        force = int(float(value))
        self.current_force = force
        self.force_value_label.config(text=f"{force}%")
        
        if self.connected and self.hand:
            self.hand.set_forces([force] * 6)
    
    def on_finger_change(self, finger_idx, value):
        """Handle finger slider change"""
        angle = int(float(value))
        self.current_angles[finger_idx] = angle
        self.finger_value_labels[finger_idx].config(text=f"{angle}%")
        
        if self.connected and self.hand:
            # Send only this finger (use -1 for others)
            angles = [-1] * 6
            angles[finger_idx] = angle
            self.hand.set_angles(angles)
    
    def open_all(self):
        """Open all fingers"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        for slider in self.finger_sliders:
            slider.set(0)
        if self.hand:
            self.hand.set_angles([0] * 6)
            self.log_feedback("Opening all fingers...")
    
    def close_all(self):
        """Close all fingers"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        for slider in self.finger_sliders:
            slider.set(100)
        if self.hand:
            self.hand.set_angles([100] * 6)
            self.log_feedback("Closing all fingers...")
    
    def half_close(self):
        """Half close all fingers"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        for slider in self.finger_sliders:
            slider.set(50)
        if self.hand:
            self.hand.set_angles([50] * 6)
            self.log_feedback("Half closing...")
    
    def reset_sliders(self):
        """Reset all sliders"""
        for slider in self.finger_sliders:
            slider.set(0)
        self.speed_slider.set(50)
        self.force_slider.set(50)
        self.log_feedback("Sliders reset")
    
    def read_angles(self):
        """Read actual angles"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        if self.hand:
            angles = self.hand.get_angles()
            if angles:
                text = "Angles: " + ", ".join([f"F{i+1}:{a:.1f}%" for i, a in enumerate(angles)])
                self.log_feedback(text)
            else:
                self.log_feedback("Failed to read angles")
    
    def read_forces(self):
        """Read actual forces"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        if self.hand:
            forces = self.hand.get_forces()
            if forces:
                text = "Forces: " + ", ".join([f"F{i+1}:{f:.1f}%" for i, f in enumerate(forces)])
                self.log_feedback(text)
            else:
                self.log_feedback("Failed to read forces")
    
    def read_temps(self):
        """Read temperatures"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please connect first")
            return
        
        if self.hand:
            temps = self.hand.get_temperatures()
            if temps:
                text = "Temps: " + ", ".join([f"F{i+1}:{t}C" for i, t in enumerate(temps)])
                self.log_feedback(text)
            else:
                self.log_feedback("Failed to read temps")
    
    def log_feedback(self, message):
        """Log message to feedback area"""
        self.feedback_text.insert(tk.END, message + "\n")
        self.feedback_text.see(tk.END)
    
    def on_closing(self):
        """Handle window closing"""
        if self.connected and self.hand:
            self.hand.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RobotHandGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

