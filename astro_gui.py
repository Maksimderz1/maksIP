import cv2
import numpy as np
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math
import os
import csv

# ================= НАСТРОЙКИ =================
MIN_AREA = 30
CROP_SIZE = 512
MAX_ZOOM = 12.0
# ============================================


# ---------- ПРЕДОБРАБОТКА ----------

def prepare_image(img):
    if img.ndim == 2:
        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[:, :, :3].astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, rgb


# ---------- РАЗБИЕНИЕ НА КРОПЫ ----------

def split_into_crops(gray):
    h, w = gray.shape
    crops = []

    for y in range(0, h, CROP_SIZE):
        for x in range(0, w, CROP_SIZE):
            crop = gray[y:y+CROP_SIZE, x:x+CROP_SIZE]
            crops.append((crop, x, y))

    return crops


# ---------- АНАЛИЗ КРОПА (ПАРАЛЛЕЛЬНО) ----------

def analyze_crop(data):
    crop, ox, oy = data
    result = []

    thresh = cv2.adaptiveThreshold(
        crop, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -4
    )

    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1, min(w, h))  # вытянутость

        mask = np.zeros(crop.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        brightness = cv2.mean(crop, mask=mask)[0]
        contrast = brightness - cv2.mean(crop)[0]

        if contrast < 10:
            continue

        # 🔥 КЛАССИФИКАЦИЯ (КОМЕТ БОЛЬШЕ)
        if area > 800 or aspect > 1.6:
            obj = "Комета"
        elif area > 200:
            obj = "Планета"
        else:
            obj = "Звезда"

        cx = ox + x + w // 2
        cy = oy + y + h // 2
        r = int(math.sqrt(area / math.pi))

        result.append({
            "type": obj,
            "x": cx,
            "y": cy,
            "r": r,
            "area": int(area),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "aspect": round(aspect, 2)
        })

    return result


def analyze_image_parallel(gray, cores):
    crops = split_into_crops(gray)
    objects = []

    with ProcessPoolExecutor(max_workers=cores) as ex:
        for res in ex.map(analyze_crop, crops):
            objects.extend(res)

    return objects


# ---------- GUI ----------

class AstroApp:
    def __init__(self, root):
        self.root = root
        root.title("Астрономический анализ (параллельный)")

        self.zoom = 1.0
        self.min_zoom = 1.0
        self.offset = [0, 0]

        # ===== верх =====
        top = tk.Frame(root)
        top.pack(fill=tk.X)

        tk.Button(top, text="Открыть TIFF", command=self.load).pack(side=tk.LEFT)
        tk.Button(top, text="Сохранить изображение", command=self.save_image).pack(side=tk.LEFT)
        tk.Button(top, text="Сохранить CSV", command=self.save_csv).pack(side=tk.LEFT)

        tk.Label(top, text="CPU").pack(side=tk.LEFT, padx=5)
        self.cores = tk.IntVar(value=multiprocessing.cpu_count())
        tk.Spinbox(top, from_=1, to=multiprocessing.cpu_count(),
                   textvariable=self.cores, width=4).pack(side=tk.LEFT)

        # ===== canvas =====
        self.canvas = tk.Canvas(root, bg="black", height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-1>", self.pick)

        # ===== инфо =====
        self.info = tk.Text(root, height=4)
        self.info.pack(fill=tk.X)

        # ===== таблица =====
        table_frame = tk.Frame(root)
        table_frame.pack(fill=tk.BOTH)

        self.table = ttk.Treeview(
            table_frame,
            columns=("type", "x", "y", "area", "brightness", "contrast", "aspect"),
            show="headings",
            height=8
        )

        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, width=110)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.base_img = None
        self.objects = []


    # ---------- ЗАГРУЗКА ----------

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff")])
        if not path:
            return

        raw = tiff.imread(path)
        gray, rgb = prepare_image(raw)

        self.objects = analyze_image_parallel(gray, self.cores.get())
        self.base_img = rgb

        self.root.update_idletasks()

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        h, w, _ = rgb.shape

        self.min_zoom = min(cw / w, ch / h)
        self.zoom = self.min_zoom
        self.offset = [
            (cw - w * self.zoom) / 2,
            (ch - h * self.zoom) / 2
        ]

        self.fill_table()
        self.draw()


    # ---------- ТАБЛИЦА ----------

    def fill_table(self):
        for r in self.table.get_children():
            self.table.delete(r)

        for o in self.objects:
            self.table.insert("", tk.END, values=(
                o["type"], o["x"], o["y"],
                o["area"], o["brightness"],
                o["contrast"], o["aspect"]
            ))


    # ---------- ОТРИСОВКА ----------

    def draw(self):
        if self.base_img is None:
            return

        self.canvas.delete("all")

        h, w, _ = self.base_img.shape
        img = cv2.resize(self.base_img, (int(w*self.zoom), int(h*self.zoom)))
        tk_img = ImageTk.PhotoImage(Image.fromarray(img))

        self.canvas.create_image(self.offset[0], self.offset[1],
                                 anchor=tk.NW, image=tk_img)
        self.canvas.image = tk_img

        colors = {
            "Звезда": "#66ffff",
            "Планета": "#ffff66",
            "Комета": "#ff6666"
        }

        for o in self.objects:
            x = o["x"] * self.zoom + self.offset[0]
            y = o["y"] * self.zoom + self.offset[1]
            r = max(2, min(5, o["r"] * self.zoom * 0.3))
            color = colors[o["type"]]

            self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                    outline=color, width=1)

            self.canvas.create_text(
                x + r + 3, y - r - 2,
                text=o["type"],
                fill=color,
                font=("Arial", 8),
                anchor=tk.NW
            )


    # ---------- КЛИК ----------

    def pick(self, e):
        for o in self.objects:
            x = o["x"] * self.zoom + self.offset[0]
            y = o["y"] * self.zoom + self.offset[1]
            if (e.x-x)**2 + (e.y-y)**2 <= (o["r"]*self.zoom)**2:
                self.info.delete("1.0", tk.END)
                self.info.insert(tk.END,
                    f"{o['type']} | X={o['x']} Y={o['y']} | "
                    f"Area={o['area']} | Bright={o['brightness']} | "
                    f"Contrast={o['contrast']} | Aspect={o['aspect']}"
                )
                break


    # ---------- СОХРАНЕНИЕ ----------

    def save_image(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        result = self.base_img.copy()
        for o in self.objects:
            color = {
                "Звезда": (0, 255, 255),
                "Планета": (255, 255, 0),
                "Комета": (255, 0, 0)
            }[o["type"]]
            cv2.circle(result, (o["x"], o["y"]), o["r"], color, 1)

        path = os.path.join(folder, "astro_parallel.png")
        cv2.imwrite(path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Готово", f"Сохранено:\n{path}")


    def save_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Тип", "X", "Y", "R", "Площадь", "Яркость", "Контраст", "Вытянутость"]
            )
            for o in self.objects:
                writer.writerow([
                    o["type"], o["x"], o["y"], o["r"],
                    o["area"], o["brightness"], o["contrast"], o["aspect"]
                ])

        messagebox.showinfo("Готово", f"CSV сохранён:\n{path}")


    def on_zoom(self, e):
        factor = 1.1 if e.delta > 0 else 0.9
        self.zoom = max(self.min_zoom, min(MAX_ZOOM, self.zoom * factor))
        self.draw()


# ---------- ЗАПУСК ----------

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x850")
    AstroApp(root)
    root.mainloop()
