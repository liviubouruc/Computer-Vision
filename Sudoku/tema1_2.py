import cv2 as cv
import numpy as np
import os


def task2(image_path):
    save_path = 'evaluare\\fisiere_solutie\\Liviu_Bouruc_334\\jigsaw'
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
        image_m_blur = cv.medianBlur(image, 5)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
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
    #show_image("img_crop", img_crop)


    img_gray = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
    image_blur = cv.medianBlur(img_gray, 7)
    #show_image("img_blur", image_blur)
    #_, thresh = cv.threshold(image_blur, 125, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(image_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 31)
    #show_image("thresh", thresh)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    #show_image("processed", thresh)


    img_bordered = thresh[5:495, 5:495]
    img_bordered = cv.copyMakeBorder(img_bordered,5,5,5,5,cv.BORDER_CONSTANT,value=(0, 0, 0))
    #show_image("bordered", img_bordered)


    cnts = cv.findContours(img_bordered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    # mask = np.zeros(thresh.shape, np.uint8)
    # for cnt in cnts:
    #     cv.drawContours(mask, cnt, -1, (255, 0, 0), 1)
    #     show_image("mask", mask)


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


    zones = [0] * len(cnts)

    def get_results(img_crop, lines_horizontal, lines_vertical):
        for j in range(len(lines_vertical)-1):
            for i in range(len(lines_horizontal)-1):
                y_min = lines_vertical[j][0][0] + 15
                y_max = lines_vertical[j+1][1][0] - 15
                x_min = lines_horizontal[i][0][1] + 15
                x_max = lines_horizontal[i+1][1][1] - 15
                patch = img_crop[y_min:y_max, x_min:x_max].copy()
            
                #print(x_min, x_max, y_min, y_max)
                #print((x_min+x_max)//2, (y_min+y_max)//2)
                #print(cv.pointPolygonTest(cnts[0], ((x_min+x_max)//2, (y_min+y_max)//2), True))

                for k in range(len(cnts)):
                    if cv.pointPolygonTest(cnts[k], ((x_min+x_max)//2, (y_min+y_max)//2), True) > 0:
                        mask = np.zeros(img_crop.shape, np.uint8)
                        cv.drawContours(mask, cnts[k], -1, (255, 0, 0), 1)
                        cv.circle(mask, ((x_min+x_max)//2, (y_min+y_max)//2), 2, (0,0,255), -1)
                        #show_image("mask", mask)
                        
                        if zones[k] != 0:
                            #print(zones[k], end='')
                            f.write(str(zones[k]))
                            fb.write(str(zones[k]))
                        else:
                            zone = max(zones)+1
                            zones[k] = zone
                            #print(zones[k], end='')
                            f.write(str(zones[k]))
                            fb.write(str(zones[k]))
                        break

                #show_image("patch", patch)
                patch_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(patch_gray, 125, 255, cv.THRESH_BINARY)
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
            if j == len(lines_vertical)-2:
                continue
            f.write('\n')
            fb.write('\n')
            

    get_results(img_crop,lines_horizontal, lines_vertical)


files = os.listdir('testare\\jigsaw\\')
for file in files:
    if file[-3:] == 'jpg':
        path = 'testare\\jigsaw\\' + file
    task2(path)
