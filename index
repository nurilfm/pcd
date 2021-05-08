import os
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msgbox
from tkinter.ttk import Treeview
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from PIL import Image
from PIL import ImageTk
import csv
import math
import random
import math
import random
import csv
import cv2
import numpy as np

import xlsxwriter
from datetime import datetime


path = ""
name = ""
status = 0
#0 : Image Selected
#1 : Grayscale
#2 :


tempList = [] #Temporary Table for exported to excel


def loadTempTable():
    # tempList.sort(key=lambda e: e[1], reverse=True)
    for i, (name, red, green, blue, contrast, dissimilarity, homogeneity, energy, asm, correlation) in enumerate(tempList, start=1):
        listBox.insert("", "end", values=(i, name, red, green, blue, contrast, dissimilarity, homogeneity, energy, asm, correlation))

def emptyTempTableView():
    listBox.delete(*listBox.get_children())

def emptyTempTable():
    global tempList
    tempList = []

    emptyTempTableView()

def addToTempTable():
    # msgbox.showinfo("ASA","ASASSAS")
    global tempList
    global name;
    if (len(path) > 0):
        # Get RGB Extraction
        r,g,b = rgb_extraction()

        # Contrast
        contrast_ = contrast()

        # Dissimilarity
        dissimilarity_ = dissimilarity()

        # Homogeneity
        homogeneity_ = homogeneity()

        # Energy
        energy_ = energy()

        # ASM
        asm_ = ASM()

        # Correlation
        correlation_ = correlation()

        #Insert to Table
        data = [name, r, g, b, contrast_, dissimilarity_, homogeneity_, energy_, asm_, correlation_,]
        tempList.append(data)
        print(tempList)

        #Empty Table
        emptyTempTableView()


        #Load Table
        loadTempTable()
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")


def select_image():
    # Variable Path Menginisialisasi Variable Global : Path
    global path
    global name

    width = 200
    height = 200
    dim = (width, height)

    path = filedialog.askopenfilename()
    folder_path, name = os.path.split(path)  # Get Name File
    print(name)
    # Jika Gambar Ada
    if (len(path) > 0):
        read_image = cv2.imread(path)
        # Picture RGB
        original = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
        # Resize Image
        resized = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
        # Set Ke Label Image
        setOriginalImage(resized)

        # Set Value RGB to TextBox
        r, g, b = rgb_extraction()

        red_value.set(r)
        green_value.set(g)
        blue_value.set(b)

        # Contrast
        contrast_ = contrast()
        contrast_value.set(contrast_)

        # Dissimiliarity
        dissimilarity_ = dissimilarity()
        dissimilarity_value.set(dissimilarity_)

        # Homogeneity
        homogeneity_ = homogeneity()
        homogeneity_value.set(homogeneity_)

        # Energy
        energy_ = energy()
        energy_value.set(energy_)

        # ASM
        ASM_ = ASM()
        ASM_value.set(ASM_)

        # Correlation
        correlation_ = correlation()
        correlation_value.set(correlation_)

#Set Original Image
def setOriginalImage(images):
    # Operasi
    images = Image.fromarray(images)
    images = ImageTk.PhotoImage(images)
    # Set Image Ke Label
    original_image.configure(image=images)
    original_image.image = images
    original_image.configure(width=200, height=200)

#set Modified Image
def setModifiedImage(images):
    # Operasi
    images = Image.fromarray(images)
    images = ImageTk.PhotoImage(images)
    # Set Image Ke Label
    modified_image.configure(image=images)
    modified_image.image = images
    modified_image.configure(width=200, height=200)

#Resize
def resize():
    width = 100
    height = 100
    dim = (width, height)
    if(len(path) > 0):
        # convert image to grayscale image
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        #Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        # Set to Label
        setModifiedImage(resized)
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")

#Grayscale
def grayscale():
    if(len(path) > 0):
        #Read Image As Grayscale
        grayscale = cv2.imread(path, 0)

        #Set to Label
        setModifiedImage(grayscale)

    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")



#NaiveBayes
def naivebayes():
    df = pd.read_csv("Naive Bayes.xlsx")
    df.head()

    '''
#Binary
def binarization():
    if(len(path) > 0):
        #Read Image As Grayscale
        grayscale = cv2.imread(path, 0)

        #Operasi Binarisasi
        (thresh, blackAndWhiteImage) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Set to Label
        setModifiedImage(blackAndWhiteImage)
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")

#Edge Detection
def edge_detection():
    if (len(path) > 0):
        #Read Image As Grayscale
        grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #Operasi Edge Detection
        edged = cv2.Canny(grayscale, 50, 100)

        # Set to Label
        setModifiedImage(edged)
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")

#Erosion
def erosion():
    if(len(path) > 0):
        #Read Image As Grayscale
        read_image = cv2.imread(path, 0)

        #Operasi Erosi
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(read_image, kernel, iterations=1)

        # Set to Label
        setModifiedImage(img_erosion)
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")
#Dilation
def dilation():
    if(len(path) > 0):
        #Read Image As Grayscale
        read_image = cv2.imread(path, 0)

        #Operasi Erosi
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(read_image, kernel, iterations=1)

        # Set to Label
        setModifiedImage(img_dilation)
    else:
        msgbox.showwarning("Warning", "No File Selected. Please Select File!")
#Perimeter
def perimeter():
    pixel_count = 0
    if(len(path) > 0):

        # Read Image As Grayscale
        img_grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Operasi Edge Detection
        edged = cv2.Canny(img_grayscale, 50, 100)

        height, width = edged.shape

        for y in range(0, height):  #Loop
            for x in range(0, width):
                color_value = edged[y][x]
                if(color_value == 255):
                    pixel_count += 1

    return pixel_count

def area():
    pixel_count = 0
    if(len(path) > 0):
        # Read Image As Grayscale
        grayscale = cv2.imread(path, 0)

        # Operasi Binarisasi
        (thresh, blackAndWhiteImage) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        height, width = blackAndWhiteImage.shape

        for y in range(0, height):  # Loop
            for x in range(0, width):
                color_value = blackAndWhiteImage[y][x]
                if (color_value == 0):
                    pixel_count += 1

    return pixel_count;



def ratio_compactness():
    val_perimeter = perimeter()
    val_area = area()

    ratio_compactness = val_perimeter * val_perimeter / val_perimeter

    return ratio_compactness
'''

#Contrast
def contrast():
    width = 200
    height = 200
    dim = (width, height)
    #pixel_count = 0
    if(len(path) > 0):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)

        # Operasi Contrast
        contrast = greycoprops(matrix_coocurrence, 'contrast')

        # Set to Label
        setModifiedImage(contrast)

    return contrast

#Dissimiliarity
def dissimilarity():
    width = 200
    height = 200
    dim = (width, height)
    if(len(path) > 0 ):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)

        # Operasi Dissimilarity
        dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')

        # Set to Label
        setModifiedImage(dissimilarity)

    return dissimilarity

##Homogeneity
def homogeneity():
    width = 200
    height = 200
    dim = (width, height)
    if(len(path) > 0 ):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)

        # Operasi Homogeneity
        homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')

        # Set to Label
        setModifiedImage(homogeneity)

    return homogeneity

#Energy
def energy():
    width = 200
    height = 200
    dim = (width, height)
    if(len(path) > 0 ):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)

        # Operasi Energy
        energy = greycoprops(matrix_coocurrence, 'energy')

        # Set to Label
        setModifiedImage(energy)

    return energy

#ASM
def ASM():
    width = 200
    height = 200
    dim = (width, height)
    if(len(path) > 0 ):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)
        # Operasi ASM
        ASM = greycoprops(matrix_coocurrence, 'ASM')

        # Set to Label
        setModifiedImage(ASM)

    return ASM

#Correlation
def correlation():
    width = 200
    height = 200
    dim = (width, height)
    if(len(path) > 0 ):
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)

        img = img_as_ubyte(resized)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(img, bins)

        # Looping for Value in Pixels
        max_value = inds.max() + 1

        # 0=0, np.pi/4=45, np.pi/2=90, 3*np.pi/4=135
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)

        # Operasi Correlation
        correlation = greycoprops(matrix_coocurrence, 'correlation')

        # Set to Label
        setModifiedImage(correlation)

    return correlation


#RGB Extraction
def rgb_extraction():
    rgb_color = []
    if(len(path) > 0):
        read_image = cv2.imread(path)
        b, g, r = cv2.split(read_image)
        # print(r, g, b)

        red_average_value = np.mean(r)
        # print(red_average_value)

        green_average_value = np.mean(g)
        # print(green_average_value)

        blue_average_value = np.mean(b)
        # print(blue_average_value)

        rgb_color = [round(red_average_value, 2), round(green_average_value, 2), round(blue_average_value, 2)]

    return rgb_color
'''
#Centroid
def centroid():
    if(len(path) > 0):
        # convert image to grayscale image
        img = cv2.imread(path)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # put text and highlight the center
        cv2.circle(original_image, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(original_image, "Centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Set to Label
        setModifiedImage(original_image)
'''

def exportExcel():
    default_excel_file_name = "Exported Data_" + datetime.now().strftime("%d_%m_%Y %H_%M_%S") + ".xlsx"
    path = filedialog.asksaveasfile(initialfile=default_excel_file_name , mode='a', title="Export Excel File", defaultextension=".xlsx",
                                    filetypes = (("Excel file","*.xlsx"),("All files","*.*")))

    if(path is not None):
        file_destination = path.name

        workbook = xlsxwriter.Workbook(file_destination)
        worksheet1 = workbook.add_worksheet()

        worksheet1.write('A1', 'No.')
        worksheet1.write('B1', 'File Name')
        worksheet1.write('C1', 'Red.')
        worksheet1.write('D1', 'Green')
        worksheet1.write('E1', 'Blue')
        worksheet1.write('F1', 'Contrast')
        worksheet1.write('G1', 'Dissimilarity')
        worksheet1.write('H1', 'Homogeneity')
        worksheet1.write('I1', 'Energy')
        worksheet1.write('J1', 'ASM')
        worksheet1.write('K1', 'Correlation')


        number = 1
        row = 2;
        for i in range(len(tempList)-1):
            worksheet1.write("A" + str(row), number)
            worksheet1.write("B" + str(row), tempList[i][0])
            worksheet1.write("C" + str(row), tempList[i][1])
            worksheet1.write("D" + str(row), tempList[i][2])
            worksheet1.write("E" + str(row), tempList[i][3])
            worksheet1.write("F" + str(row), tempList[i][4])
            worksheet1.write("G" + str(row), tempList[i][5])
            worksheet1.write("H" + str(row), tempList[i][6])
            worksheet1.write("I" + str(row), tempList[i][7])
            worksheet1.write("J" + str(row), tempList[i][8])
            worksheet1.write("K" + str(row), tempList[i][9])

            number += 1
            row += 1

            def f(x):
                return np.int(x)

            f2 = np.vectorize(f)
            x = np.arange(1, 15.1, 0.1)
            plt.plot(x, f2(x))
            plt.show()

        try:
            workbook.close()
            msgbox.showinfo("Success", "Export Excel File Success")
        except:
            msgbox.showerror("Success", "Export Excel File Failed")


##### Tkinter Design ####

root = Tk()
root.title("Skripsi")
root.geometry("1500x950")

app_title = Label(root, text="Nuril Feby Maulidyah - Nim E41170153 || IDENTIFIKASI KUALITAS SUSU SAPI PERAH", font="Times 12 bold")
app_title.pack(pady=10)
#Centered Width and Height of Apps in Monitor
app_width = 1200
app_height = 650
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (app_width / 2)
y = (screen_height / 2) - (app_height / 2)

root.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")

txt_original_image = Label(root, text="Original Image")
txt_original_image.place(x=10, y=35)

txt_modified_image = Label(root, text="Modified Image")
txt_modified_image.place(x=250, y=35)

original_image = Label(root)
original_image.place(x=10, y=70)

modified_image = Label(root)
# modified_image.configure(width=200, height=200)
modified_image.place(x=250, y=70)

btnSelectImage = Button(root, text="Select Image", command=select_image, width=15)
btnSelectImage.place(x=480, y=70)
#
btnGrayscale = Button(root, text="Grayscale", command=grayscale, width=15)
btnGrayscale.place(x=630, y=70)
#
btnResize = Button(root, text="Resize", command=resize,  width=15)
btnResize.place(x=480, y=105)
#
btnNaiveBayes = Button(root, text="Naive Bayes", command=naive_bayes, width=15)
btnNaiveBayes.place(x=630, y=105)


'''
btnBinarization = Button(root, text="Binarization", command=binarization,  width=15)
btnBinarization.place(x=480, y=105)
#
btnEdgeDetection = Button(root, text="Edge Detection", command=edge_detection,  width=15)
btnEdgeDetection.place(x=630, y=105)
#
btnErosion = Button(root, text="Erosion", command=erosion, width=15)
btnErosion.place(x=480, y=140)
#
btnDilation = Button(root, text="Dilation", command=dilation, width=15)
btnDilation.place(x=630, y=140)

btnCentroid = Button(root, text="Centroid", command=centroid, width=15)
btnCentroid.place(x=480, y=175)
'''

#Label For Text Box

label_red_value = Label(root, text="Red Value")
label_red_value.place(x=10, y=318)

label_green_value = Label(root, text="Green Value")
label_green_value.place(x=120, y=318)

label_blue_value = Label(root, text="Blue Value")
label_blue_value.place(x=230, y=318)

label_contrast = Label(root, text="Contrast")
label_contrast.place(x=340, y=318) # awal 388, y - 80

label_dissimilarity_value = Label(root, text="Dissimilarity")
label_dissimilarity_value.place(x=450, y=318)  #35

label_homogeneity_value = Label(root, text="Homogeneity")
label_homogeneity_value.place(x=560, y=318)

label_energy_value = Label(root, text="Energy")
label_energy_value.place(x=680, y=318)

label_ASM_value = Label(root, text="ASM")
label_ASM_value.place(x=780, y=318)

label_correlation_value = Label(root, text="Correlation")
label_correlation_value.place(x=880, y=318)

#Text Box with images extraction value
red_value = DoubleVar()
txt_red_value = Entry(root, width=15, textvariable=red_value, state="readonly")
txt_red_value.place(x=10, y=345)

green_value = DoubleVar()
txt_green_value = Entry(root, width=15, textvariable=green_value, state="readonly")
txt_green_value.place(x=120, y=345)

blue_value = DoubleVar()
txt_blue_value = Entry(root, width=15, textvariable=blue_value, state="readonly")
txt_blue_value.place(x=230, y=345)

#Text Box with GLCM

contrast_value = IntVar()
txt_contrast_value = Entry(root, width=15, textvariable=contrast_value, state="readonly")
txt_contrast_value.place(x=340, y=345) #388+27 =415

dissimilarity_value = IntVar()
txt_dissimilarity_value = Entry(root, width=15, textvariable=dissimilarity_value, state="readonly")
txt_dissimilarity_value.place(x=450, y=345)

homogeneity_value = DoubleVar()
txt_homogeneity_value = Entry(root, width=15, textvariable=homogeneity_value, state="readonly")
txt_homogeneity_value.place(x=560, y=345)

energy_value = DoubleVar()
txt_energy_value = Entry(root, width=15, textvariable=energy_value, state="readonly")
txt_energy_value.place(x=665, y=345)

ASM_value = DoubleVar()
txt_ASM_value = Entry(root, width=15, textvariable=ASM_value, state="readonly")
txt_ASM_value.place(x=770, y=345)

correlation_value = DoubleVar()
txt_correlation_value = Entry(root, width=15, textvariable=correlation_value, state="readonly")
txt_correlation_value.place(x=875, y=345)

#Button Add Extraction values To Table
btnAddToListExportExcel = Button(root, text="Add to List Export Excel", width=20, command=addToTempTable)
btnAddToListExportExcel.place(x=10, y=375)

#Label For Table

label_table_list_exported = Label(root, text="List Export To Excel")
label_table_list_exported.place(x=10, y=405)

#Add Frame for grid. If we not define new frame, app will crash because cannot mix grid and place method
frame_for_grid = Frame(root)
frame_for_grid.place(x=10, y=435)

#Table Temporary to excel
cols = ('No.', 'Name', 'R', 'G', 'B', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'ASM', 'Correlation')
listBox = Treeview(frame_for_grid, columns=cols, show='headings',)

listBox.column(0, width=40, stretch=FALSE)
listBox.column(1, width=200, stretch=FALSE)
listBox.column(2, width=100, stretch=FALSE)
listBox.column(3, width=100, stretch=FALSE)
listBox.column(4, width=100, stretch=FALSE)
listBox.column(5, width=125, stretch=FALSE)
listBox.column(6, width=125, stretch=FALSE)
listBox.column(7, width=125, stretch=FALSE)
listBox.column(8, width=125, stretch=FALSE)
listBox.column(9, width=125, stretch=FALSE)
listBox.column(10, width=125, stretch=FALSE)

for col in cols:
    listBox.heading(col, text=col)
listBox.grid(row=1, column=0, columnspan=2)

#Scrollbar
scrollbar = Scrollbar(root, orient="vertical", command=listBox.yview)
scrollbar.pack(side='right', fill='y')

listBox.configure(yscrollcommand=scrollbar.set)

#Empty Row Button
btnEmptyTempTable = Button(root, text="Empty Row", width=15, command=emptyTempTable)
btnEmptyTempTable.place(x=10, y=685)

#Export Excel Button
btnExportExcel = Button(root, text="Export Excel", width=15, command=exportExcel)
btnExportExcel.place(x=180, y=685)

#Kick Off Lur
root.mainloop()
