import cv2 as cv
import numpy as np
import os


def task1(image_path):
    save_path = 'evaluare\\fisiere_solutie\\Liviu_Bouruc_334\\clasic'
    file_name = os.path.join(save_path, image_path[-6:-4] + '_predicted.txt')
    file_name_bonus = os.path.join(save_path, image_path[-6:-4] + '_bonus_predicted.txt')
    f = open(file_name, 'w+')
    fb = open(file_name_bonus, 'w+')


    def show_image(title, image):
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def preprocess_image(image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_m_blur = cv.medianBlur(image, 3)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 3) 
        image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
        _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.erode(thresh, kernel)
        
        # show_image("median blurred", image_m_blur)
        # show_image("gaussian blurred", image_g_blur)
        # show_image("sharpened", image_sharpened)    
        # show_image("threshold of blur", thresh)
        
        edges = cv.Canny(thresh, 150, 400)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
    
    
        for i in range(len(contours)):
            if(len(contours[i]) > 3):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1]:
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis=1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left
        
        image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
        cv.circle(image_copy, tuple(top_left), 4, (0,0,255), -1)
        cv.circle(image_copy, tuple(top_right), 4, (0,0,255), -1)
        cv.circle(image_copy, tuple(bottom_left), 4, (0,0,255), -1)
        cv.circle(image_copy, tuple(bottom_right), 4, (0,0,255), -1)
        #show_image("detected corners", image_copy)
        
        return top_left,top_right,bottom_left,bottom_right


    img = cv.imread(image_path)
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    result = preprocess_image(img)

    pts1 = np.float32([result[0], result[1], result[2], result[3]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_crop = cv.warpPerspective(img, matrix, (500, 500))
    #show_image("perspcropped", img_crop)


    lines_vertical = []
    for i in range(0, 500, 55):
        l = []
        l.append((i, 0))
        l.append((i, 499))
        lines_vertical.append(l)

    lines_horizontal = []
    for i in range(0, 500, 55):
        l = []
        l.append((0, i))
        l.append((499, i))
        lines_horizontal.append(l)

    for line in lines_vertical: 
        cv.line(img_crop, line[0], line[1], (0, 255, 0), 5)
    for line in lines_horizontal: 
        cv.line(img_crop, line[0], line[1], (0, 0, 255), 5)
    #show_image("img", img_crop)


    def get_results(img_crop, lines_horizontal, lines_vertical):
        for i in range(len(lines_horizontal)-1):
            for j in range(len(lines_vertical)-1):
                y_min = lines_vertical[j][0][0] + 15
                y_max = lines_vertical[j+1][1][0] - 15
                x_min = lines_horizontal[i][0][1] + 15
                x_max = lines_horizontal[i+1][1][1] - 15
                patch = img_crop[x_min:x_max, y_min:y_max].copy()
                #show_image("patch", patch)

                _, thresh = cv.threshold(patch, 150, 255, cv.THRESH_BINARY)
                #show_image("patch_th", thresh)
                if 0 in thresh:
                    #print("x", end='')
                    f.write("x")
                    fb.write("x")
                else:
                    #print("o", end='')
                    f.write("o")
                    fb.write("o")
            #print()
            if i == len(lines_horizontal)-2:
                continue
            f.write('\n')
            fb.write('\n')

    get_results(img_crop,lines_horizontal, lines_vertical)

    f.close()
    fb.close()


files = os.listdir('.\\testare\\clasic\\')
for file in files:
    if file[-3:] == 'jpg':
        path = '.\\testare\\clasic\\' + file
    task1(path)