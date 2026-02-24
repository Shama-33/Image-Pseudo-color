import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import Global_Pseudo
import  numpy as np
import Local_hist
import glob_obj
import roi_diff

# Global variables to store selected images
input_img = None
reference_img = None
input_img_path = None
reference_img_path = None

def select_input_image():
    global input_img,input_img_path
    input_path = filedialog.askopenfilename(
        title="Select Input Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if input_path:
        try:
            input_img = cv2.imread(input_path , cv2.IMREAD_GRAYSCALE)
            input_img = cv2.resize(input_img, (512, 512))
            #input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_path = input_path
            display_input_image(input_img)
            check_show_pseudocolor_buttons()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process input image: {e}")
    else:
        messagebox.showwarning("No Selection", "No input image was selected.")

def select_reference_image():
    global reference_img,reference_img_path
    reference_path = filedialog.askopenfilename(
        title="Select Reference Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if reference_path:
        try:
            reference_img = cv2.imread(reference_path)
            reference_img = cv2.resize(reference_img, (512, 512))
            reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            reference_img_path = reference_path
            display_reference_image(reference_img)
            check_show_pseudocolor_buttons()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process reference image: {e}")
    else:
        messagebox.showwarning("No Selection", "No reference image was selected.")

def display_input_image(image):
    input_tk = cv2_to_photoimage(image)
    input_label.config(image=input_tk)
    input_label.image = input_tk

def display_reference_image(image):
    reference_tk = cv2_to_photoimage(image)
    reference_label.config(image=reference_tk)
    reference_label.image = reference_tk

def check_show_pseudocolor_buttons():
    if reference_img is not None and input_img is not None:
        false_color_label.pack(pady=10)
        local_button.pack(pady=5)
        global_button.pack(pady=5)
        local_button2.pack(pady=5)

def local_pseudocolor():
    if input_img_path and reference_img_path:
        Local_hist.main(input_img_path, reference_img_path)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def global_pseudocolor():
    #messagebox.showinfo("Global Pseudocolor", "Global pseudocoloring applied.")
    if input_img_path and reference_img_path:
        #Global_Pseudo.apply_global_pseudocolor(input_img_path, reference_img_path)
        Global_Pseudo.object_based(input_img_path, reference_img_path)
        Global_Pseudo.Histogram(input_img_path, reference_img_path)
        #Global_Pseudo.main(input_img_path, reference_img_path)
        glob_obj.main(input_img_path,reference_img_path)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def local_pseudocolor_different():
    if input_img_path and reference_img_path:
        roi_diff.main(input_img_path, reference_img_path)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def cv2_to_photoimage(cv2_image):
    return tk.PhotoImage(data=cv2.imencode('.png', cv2_image)[1].tobytes())

# Create the main window
root = tk.Tk()
root.title("PseudoColor")
root.geometry("1100x800")

# Create a canvas and a scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Add a frame for the top section (image selection and display)
top_frame = tk.Frame(scrollable_frame)
top_frame.pack(side="top", fill="x", pady=10)

# Add buttons to select input and reference images
button_style = {'font': ('Arial', 14), 'bg': 'blue', 'fg': 'white', 'padx': 20, 'pady': 10}
button_style2 = {'font': ('Arial', 14), 'bg': 'light blue', 'fg': 'black', 'padx': 20, 'pady': 10}

input_button = tk.Button(top_frame, text="Select Input Image", command=select_input_image, **button_style)
input_button.pack(side=tk.LEFT, padx=10)

reference_button = tk.Button(top_frame, text="Select Reference Image", command=select_reference_image, **button_style2)
reference_button.pack(side=tk.LEFT, padx=10)

# Labels to display the input and reference images
image_frame = tk.Frame(scrollable_frame)
image_frame.pack(side="top", pady=20)

input_label = tk.Label(image_frame)
input_label.pack(side=tk.LEFT, padx=10)

reference_label = tk.Label(image_frame)
reference_label.pack(side=tk.LEFT, padx=10)

# Label to display "False Color Your Input Image" after both images are selected
false_color_label = tk.Label(scrollable_frame, text="False Color Your Input Image", font=("Arial", 16))


# Buttons for local and global pseudocoloring (initially hidden)
local_button = tk.Button(scrollable_frame, text="Local Pseudocolor(Same Region)", command=local_pseudocolor, **button_style)
global_button = tk.Button(scrollable_frame, text="Global Pseudocolor", command=global_pseudocolor, **button_style2)
local_button2= tk.Button(scrollable_frame, text="Local Pseudocolor(Different Region)", command=local_pseudocolor_different, **button_style)

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")



# Run the application
root.mainloop()
