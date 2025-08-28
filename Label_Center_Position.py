import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import tifffile as tiff
import numpy as np
import os
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D TIF Center Annotation Tool")

        self.image_data = None
        self.image_path = None
        self.annotations = []
        self.unconfirmed_point = None
        self.last_click_coords = None

        self.current_line = 1
        self.current_column = 1

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.load_button = tk.Button(button_frame, text="📂 Load 3D TIF", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(button_frame, text="💾 Save CSV", command=self.save_csv, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)

        self.new_line_button = tk.Button(button_frame, text="➕ New Line", command=self.new_line, state=tk.DISABLED)
        self.new_line_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(button_frame, text="🗑️ Delete Last", command=self.delete_last_annotation, state=tk.DISABLED)
        self.delete_button.pack(side=tk.LEFT)

        self.status_label = tk.Label(button_frame, text="", anchor="w")
        self.status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.root.bind("<Return>", self.confirm_label)
        self.root.bind("<Escape>", self.reset_view)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIF files", "*.tif *.tiff")])
        if not file_path:
            return

        self.image_data = tiff.imread(file_path)
        if self.image_data.ndim != 3:
            messagebox.showerror("Error", "Selected file is not a 3D TIF.")
            return

        self.image_path = file_path
        self.annotations = []
        self.unconfirmed_point = None
        self.last_click_coords = None

        self.current_line = 1
        self.current_column = 1

        self.mid_idx = self.image_data.shape[0] // 2
        self.slice = self.image_data[self.mid_idx, :, :]

        self.ax.clear()
        self.ax.imshow(self.slice, cmap="gray")
        self.ax.set_title(f"Middle Slice: {self.mid_idx} — {os.path.basename(file_path)}")
        self.ax.axis("off")
        self.canvas.draw()

        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

        csv_path = self.get_csv_path()
        if os.path.exists(csv_path):
            self.load_csv(csv_path)

        self.save_button.config(state=tk.NORMAL)
        self.delete_button.config(state=tk.NORMAL if self.annotations else tk.DISABLED)
        self.new_line_button.config(state=tk.NORMAL)
        self.update_status()

    def get_csv_path(self):
        base, _ = os.path.splitext(self.image_path)
        return f"{base}_labels.csv"

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left-click
            if self.unconfirmed_point:
                self.unconfirmed_point.remove()
                self.unconfirmed_point = None

            self.unconfirmed_point = self.ax.plot(x, y, 'ro', markersize=6)[0]
            self.last_click_coords = (x, y)
            self.canvas.draw()
            self.status_label.config(text=f"Line {self.current_line}, Column {self.current_column} — Right-click or press Enter to confirm")

        elif event.button == 3:  # Right-click
            if self.unconfirmed_point and self.last_click_coords:
                self.prompt_for_label(*self.last_click_coords)

    def confirm_label(self, event=None):
        if self.unconfirmed_point and self.last_click_coords:
            self.prompt_for_label(*self.last_click_coords)

    def prompt_for_label(self, x, y):
        base_label = simpledialog.askstring("Label", f"Enter label for L{self.current_line}C{self.current_column}:")
        if base_label:
            full_label = f"L{self.current_line}_C{self.current_column}__{base_label}"
            self.annotations.append((x, y, full_label))
            self.ax.plot(x, y, 'go', markersize=6)
            self.ax.text(x + 5, y, full_label, color='red', fontsize=14, weight='bold')
            if self.unconfirmed_point:
                self.unconfirmed_point.remove()
            self.unconfirmed_point = None
            self.last_click_coords = None
            self.current_column += 1

            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)

            self.canvas.draw()
            self.delete_button.config(state=tk.NORMAL)
            self.update_status()
        else:
            self.status_label.config(text="Labeling cancelled.")

    def new_line(self):
        self.current_line += 1
        self.current_column = 1
        self.update_status()

    def delete_last_annotation(self):
        if not self.annotations:
            return

        deleted_annotation = self.annotations.pop()
        self.redraw_annotations()

        if self.annotations:
            last_label = self.annotations[-1][2]
            try:
                parts = last_label.split("__")[0]
                line_col = parts.split("_")
                self.current_line = int(line_col[0][1:])
                self.current_column = int(line_col[1][1:]) + 1
            except Exception:
                self.current_line = 1
                self.current_column = 1
        else:
            self.current_line = 1
            self.current_column = 1
            self.delete_button.config(state=tk.DISABLED)

        self.status_label.config(text=f"Deleted: {deleted_annotation[2]}")
        self.update_status()

    def redraw_annotations(self):
        self.ax.clear()
        self.ax.imshow(self.slice, cmap="gray")
        self.ax.set_title(f"Middle Slice: {self.mid_idx} — {os.path.basename(self.image_path)}")
        self.ax.axis("off")

        for x, y, label in self.annotations:
            self.ax.plot(x, y, 'go', markersize=6)
            self.ax.text(x + 5, y, label, color='red', fontsize=14, weight='bold')

        self.canvas.draw()

    def on_scroll(self, event):
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
        self.ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])
        self.canvas.draw()

    def reset_view(self, event=None):
        if hasattr(self, 'initial_xlim') and hasattr(self, 'initial_ylim'):
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
            self.canvas.draw()
            self.status_label.config(text="View reset.")

    def save_csv(self):
        if not self.image_path:
            return

        csv_path = self.get_csv_path()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "label"])
            for x, y, label in self.annotations:
                writer.writerow([x, y, self.mid_idx, label])

        self.status_label.config(text=f"Saved to: {os.path.basename(csv_path)}")

    def load_csv(self, path):
        try:
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x, y, label = int(row['x']), int(row['y']), row['label']
                    self.annotations.append((x, y, label))
                    self.ax.plot(x, y, 'go', markersize=6)
                    self.ax.text(x + 5, y, label, color='red', fontsize=14, weight='bold')

            # Resume from last label
            if self.annotations:
                last_label = self.annotations[-1][2]
                try:
                    parts = last_label.split("__")[0]
                    line_col = parts.split("_")
                    self.current_line = int(line_col[0][1:])
                    self.current_column = int(line_col[1][1:]) + 1
                except Exception:
                    self.current_line = 1
                    self.current_column = 1

            self.canvas.draw()
            self.status_label.config(text="Loaded existing annotations.")
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))

    def update_status(self):
        self.status_label.config(
            text=f"Line: {self.current_line}, Column: {self.current_column} — Left-click to add, Right-click or Enter to label"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
