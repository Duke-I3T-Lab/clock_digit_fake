import cv2
import numpy as np


"""
Dashes index: 

--- 5 ---
|       |
1       2
|       |
--- 6 ---
|       |
3       4
|       |
--- 7 ---
So that any index i will be identical to index i+2
"""

class DigitFaker:
    DASH_INDEX_POS_DICT = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1), 5: (0, 0), 6: (1, 0), 7: (1, 1)}
    DIGIT_CONTAIN_DASH_INDEX = {1: {2, 4}, 2: {5, 2, 6, 3, 7}, 3: {5, 2, 6, 4, 7}, 4: {1, 6, 2, 4}, 5: {5, 1, 6, 4, 7}, 6: {5, 1, 6, 3, 4, 7}, 7: {5, 2, 4}, 8: {1, 2, 3, 4, 5, 6, 7}, 9: {1, 2, 4, 5, 6, 7}, 0: {1, 2, 3, 4, 5, 7}}
    
    def __init__(self, digit_detection_method="easyocr", texture_match="sam", blending="gaussian", debug=False):
        self.digit_detection_method = digit_detection_method
        self.texture_match = texture_match
        self.blending = blending
        self.digits = None
        self.debug = debug

    def find_all_possible_changes(self, digit, dash_count_max=2):
        super_set_digits = [d for d in self.DIGIT_CONTAIN_DASH_INDEX if self.DIGIT_CONTAIN_DASH_INDEX[d].issuperset(self.DIGIT_CONTAIN_DASH_INDEX[digit]) and d != digit]
        print(super_set_digits)
        return None

    def fake_digit(self, img_path):
        image = cv2.imread(img_path)
        self.image = image
        digit_box_dict = self.detect_digits(image)
        print(digit_box_dict)
        digit = 2
        box = digit_box_dict[str(digit)]
        self.extract_pixels_from_box(box, digit)

    def detect_digits(self, image):
        bounding_boxes = {}
        if self.digit_detection_method == "easyocr":   
            import easyocr
            reader = easyocr.Reader(['en'])  # Add additional languages if needed

            # Perform OCR on the image
            results = reader.readtext(image, allowlist='0123456789')

            # Iterate through the OCR results and draw bounding boxes
            for (bbox, text, confidence) in sorted(results, key=lambda x: x[0][0][0]):
                if confidence > 0.6:  # Confidence threshold
                    bounding_boxes[eval(text)] = bbox
        elif self.digit_detection_method == "hand":
            bounding_boxes["2"] = [[175, 840], [376, 840], [376, 1099], [175, 1099]]
        elif self.digit_detection_method == "contour":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 1)
            # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 , 2
            )
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            if self.debug: # show thresh
                cv2.imshow('thresh', thresh)
                cv2.waitKey(0)


            # cv2.imshow('thresh', thresh)
            # cv2.waitKey(0)
            # detect edges and show
            # edges = cv2.Canny(thresh, 100, 200)
            # kernel = np.ones((3, 3), np.uint8)
            
            # _, edges = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)
            # # show edges
            # cv2.imshow('edges', edges)
            # cv2.waitKey(0)
            

            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if self.debug:
                contour_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                # show the image with the drawn contours
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 5 < w and 5 < h :
                        cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('contour image', contour_image)
                cv2.waitKey(0)

            contours = self.filter_opencv_contours(contours, image)
            self.wrapped_image, self.inv_transform_matrix = self.converting_contours_to_digits(contours, image)
            self.available_dashes = self.find_all_resource()
            if self.debug:
                cv2.destroyAllWindows()
            self.interactive_clock_modification()
            
            
            

        return bounding_boxes 

    def find_all_resource(self):
        assert self.digits is not None
        available_dashes = {}
        for i, digit in enumerate(self.digits):
            for dash in self.DIGIT_CONTAIN_DASH_INDEX[digit]:
                available_dashes[dash] = available_dashes.get(dash, []) + [i]
        return available_dashes

    def interactive_clock_modification(self):
        step = 1  # Track the current step
        digit_index = -1  # Selected digit index
        action = None  # Chosen action (add or remove)

        while True:
            if step == 1:
                # Step 1: Display digits and prompt the user to choose
                print("\nCurrent Clock Digits:")
                for i, digit in enumerate(self.digits):
                    print(f"Digit {i+1}: Active digit = {digit}")
                
                print("\nWhich digit would you like to modify?")
                print("Enter a number (1 to {}) or 'q' to quit:".format(len(self.digits)))

                user_input = input("> ").strip()
                if user_input.lower() == 'q':
                    print("Exiting...")
                    break
                elif user_input.isdigit() and 1 <= int(user_input) <= len(self.digits):
                    digit_index = int(user_input) - 1
                    step = 2
                else:
                    print("Invalid input. Please try again.")

            elif step == 2:
                # Step 2: Ask the user whether to add or remove dashes
                print(f"\nSelected Digit {digit_index+1}: Active digit = {self.digits[digit_index]}")
                print("Do you want to 'add' or 'remove' dashes? (Type 'back' to go back)")

                user_input = input("> ").strip().lower()
                if user_input == 'back':
                    step = 1
                elif user_input in ['add', 'remove']:
                    action = user_input
                    step = 3
                else:
                    print("Invalid input. Please try again.")

            elif step == 3:
                # Step 3: Present the user with options for modification
                print(f"\nYou chose to {action} dashes for Digit {digit_index+1}:")
                print(f"Active digit: {self.digits[digit_index]}")

                if action == 'add':
                    print("Here are the available optionals for addition:")
                    # Determine missing dashes (assuming 1-7 are possible)
                    addition_options = self.find_add_options(digit_index)
                    for new_digit in addition_options:
                        print(f"Add dashes {addition_options[new_digit]} to get {new_digit}")
                else:
                    print("Here are the available options for removal:")
                    removal_options = self.find_remove_options(digit_index)
                    for new_digit in removal_options:
                        print(f"Remove dashes {removal_options[new_digit]} to get {new_digit}")


                print("Enter the digit you want to change to, or type 'back' to go back:")
                user_input = input("> ").strip().lower()

                if user_input == 'back':
                    step = 2
                elif user_input.isdigit():
                    new_digit = int(user_input)
                    if action == 'add' and new_digit in addition_options:
                        modified_image, pad_offset_mat = self.add_dashes(self.wrapped_image, digit_index, addition_options[new_digit])
                        full_modified_image = self.inv_to_original_image(self.image, modified_image, self.inv_transform_matrix, pad_offset_mat)
                        cv2.imshow("original image", self.image)
                        cv2.imshow('full modified image', full_modified_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    
                    elif action == 'remove' and new_digit in removal_options:
                        modified_image = self.hide_dashes(self.wrapped_image, digit_index, removal_options[new_digit])
                        full_modified_image = self.inv_to_original_image(self.image, modified_image, self.inv_transform_matrix)
                        cv2.imshow("original image", self.image)
                        cv2.imshow('full modified image', full_modified_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        # self.hide_dashes_static(self.wrapped_image, digit_index, removal_options[new_digit])

                    else:
                        print("Invalid dash selection. Please try again.")
                else:
                    print("Invalid input. Please try again.")

    def find_remove_options(self, digit_index):
        digit = self.digits[digit_index]
        dashes = self.DIGIT_CONTAIN_DASH_INDEX[digit]
        # find all possible digits that can be obtained be removing dashes
        possible_removals = {}
        for sub_digit in self.DIGIT_CONTAIN_DASH_INDEX:
            if sub_digit != digit:
                if self.DIGIT_CONTAIN_DASH_INDEX[sub_digit].issubset(dashes):
                    possible_removals[sub_digit] = dashes.difference(self.DIGIT_CONTAIN_DASH_INDEX[sub_digit])
                   
        return possible_removals

    def find_add_options(self, digit_index):
        digit = self.digits[digit_index]
        dashes = self.DIGIT_CONTAIN_DASH_INDEX[digit]
        # find all possible digits that can be obtained be adding dashes
        possible_additions = {}
        for sup_digit in self.DIGIT_CONTAIN_DASH_INDEX:
            if sup_digit != digit:
                if self.DIGIT_CONTAIN_DASH_INDEX[sup_digit].issuperset(dashes):
                    needed_dashes = self.DIGIT_CONTAIN_DASH_INDEX[sup_digit].difference(dashes)
                    if set(self.available_dashes.keys()).issuperset(needed_dashes):
                        possible_additions[sup_digit] = needed_dashes
        print("possible_additions: ", possible_additions)
        return possible_additions

    def filter_opencv_contours(self, contours, image):
        filtered_contours = []
        full_height, full_width = image.shape[:2]
        min_rect_height = full_height // 7
        max_rect_height = full_height // 3
        min_rect_width = full_width // 10
        max_rect_width = full_width // 5
        min_aspect_ratio = 0.15
        max_aspect_ratio = 0.4
        prev_bbox = [0, 0, 0, 0]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if [x, y, w, h] == prev_bbox:
                continue
            prev_bbox = [x, y, w, h]
            # filtering logic. Can be similarity based, if we know the clock. see `cv2.matchShapes()`
            # contour_solidity = cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour))
            if cv2.contourArea(cv2.convexHull(contour)) == 0 or cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)) < 0.8:
                continue
            if w >= h and (w < min_rect_width or h/w < min_aspect_ratio)or h >= w and (h < min_rect_height or w/h < min_aspect_ratio)  or w > max_rect_width or h > max_rect_height: 
                continue
            elif x <= 0 or y <= 0 or x+w >= full_width or y+h >= full_height: 
                continue
            else:
                filtered_contours.append(contour)
            # prev_bbox = [x, y, w, h]
        return filtered_contours
    
    def converting_contours_to_digits(self, contours, image):
        """
        given the contours, we can separate 4 digits based on the right-most coordinate. For the same digit, they should be no more than width // 20 away, but for new digit they should be at least width//10 away
        """
        # sort the contours based on the right-most coordinate
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        all_big_x_width = [cv2.boundingRect(contour)[2] for contour in contours if cv2.boundingRect(contour)[2] > cv2.boundingRect(contour)[3] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_big_x_width = sum(all_big_x_width) // len(all_big_x_width)
        all_small_x_width = [cv2.boundingRect(contour)[2] for contour in contours if cv2.boundingRect(contour)[3] > cv2.boundingRect(contour)[2] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_small_x_width = sum(all_small_x_width) // len(all_small_x_width)

        digit_contours = [[contours[0]]]
        for i in range(1, len(contours)):
            # show contour i-1

            x1, y1, w1, h1 = cv2.boundingRect(contours[i-1])
            # cv2.drawContours(image, [contours[i-1]], -1, (0, 255, 0), 2)
            # cv2.imshow('image', image)   
            # cv2.waitKey(0)
            # im = image.copy()
            

            x2, y2, w2, h2 = cv2.boundingRect(contours[i])
            r1 = x1 + w1
            r2 = x2 + w2

            if x2 < r1 or x2 - x1 <= self.avg_small_x_width * 0.5 or self.avg_big_x_width * 0.8 < x2 - x1 < self.avg_big_x_width * 1.2 and r2 - r1 <= self.avg_big_x_width * 1.2:
                digit_contours[-1].append(contours[i])
            else:
                  digit_contours.append([contours[i]])
        self.digits, self.dashes_dicts = [], []   

        warpped_image, new_digit_contours, inv_transform_matrix = self.perspective_transform(image, digit_contours)
        # show new digit contours
        # for i, digit_contour in enumerate(new_digit_contours):
        #     for contour in digit_contour:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cv2.rectangle(warped_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow('warped_image', warped_image)

        flattened_contours = [contour for digit_contour in new_digit_contours for contour in digit_contour]
        contours = sorted(flattened_contours, key=lambda x: cv2.boundingRect(x)[0])
        self.all_y_max = max([cv2.boundingRect(contour)[1] for contour in contours])
        self.all_y_min = min([cv2.boundingRect(contour)[1] for contour in contours])
        all_big_x_width = [cv2.boundingRect(contour)[2] for contour in contours if cv2.boundingRect(contour)[2] > cv2.boundingRect(contour)[3] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_big_x_width = sum(all_big_x_width) // len(all_big_x_width)
        all_small_x_width = [cv2.boundingRect(contour)[2] for contour in contours if cv2.boundingRect(contour)[3] > cv2.boundingRect(contour)[2] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_small_x_width = sum(all_small_x_width) // len(all_small_x_width)

        all_big_y_height = [cv2.boundingRect(contour)[3] for contour in contours if cv2.boundingRect(contour)[3] > cv2.boundingRect(contour)[2] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_big_y_height = sum(all_big_y_height) // len(all_big_y_height)
        all_small_y_height = [cv2.boundingRect(contour)[3] for contour in contours if cv2.boundingRect(contour)[2] > cv2.boundingRect(contour)[3] and compute_contour_aspect_ratio(contour) < 0.5]
        self.avg_small_y_height = sum(all_small_y_height) // len(all_small_y_height)

        for i, digit_contour in enumerate(new_digit_contours):
            recognized_digit, dashes_dict = self.compute_digit_from_contours(digit_contour, warpped_image)
            self.digits.append(recognized_digit)
            self.dashes_dicts.append(dashes_dict)
        print(f"Recognized time: {self.digits[0]}{self.digits[1]}:{self.digits[2]}{self.digits[3]}")
        return warpped_image, inv_transform_matrix
    
        # self.hide_dashes_static(image, 3, 3)
        # self.hide_dashes(image, 3, 1)

    def find_four_corners(self, four_contours, margin=0):
        left_contours, right_contours = four_contours[0], four_contours[-1]
        left_most = min([cv2.boundingRect(contour)[0] for contour in left_contours]) - margin
        left_top = min([cv2.boundingRect(contour)[1] for contour in left_contours]) - margin
        left_bottom = max([cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in left_contours]) + margin
        right_most = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in right_contours]) + margin
        right_top = min([cv2.boundingRect(contour)[1] for contour in right_contours]) - margin
        right_bottom = max([cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in right_contours]) + margin
        return [(left_most, left_top), (right_most, right_top), (right_most, right_bottom), (left_most, left_bottom)]
        
    def perspective_transform(self, image, four_contours, output_width=1300, output_height=320):
        # Define the points for the output perspective transform

        corners = self.find_four_corners(four_contours, margin=3)
        
        dst_points = np.array([
            [300, 0],  # Top-left corner of the output
            [output_width - 1, 0],  # Top-right corner of the output
            [output_width - 1, output_height - 1],  # Bottom-right corner of the output
            [300, output_height - 1]  # Bottom-left corner of the output
        ], dtype="float32")

        # Prepare the source points for perspective transform
        src_points = np.array(corners, dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        def transform_contours(contours, matrix):
            transformed_contours = []
            for contour in contours:
                # Reshape contour points into (N, 1, 2)
                points = contour.reshape(-1, 1, 2).astype(np.float32)
                # Apply perspective transformation
                transformed_points = cv2.perspectiveTransform(points, matrix)
                transformed_contours.append(transformed_points)
            return transformed_contours

        # Perform the perspective warp
        warped = cv2.warpPerspective(image, M, (output_width, output_height))

        '''
        # Display the result
        # Extract the region of interest using the four corners
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(corners)], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Crop the bounding box around the four corners
        x, y, w, h = cv2.boundingRect(np.float32(corners))
        cropped_image = masked_image[y:y+h, x:x+w]

        cv2.imshow('Original Image', cropped_image)
        cv2.imshow('Warped Image', warped)

        cv2.rectangle(warped, (50, 50), (100, 100), (0, 0, 255), -1)
        warped_back = cv2.warpPerspective(warped, M_inv, (image.shape[1], image.shape[0]))
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [src_points.astype(np.int32)], (255, 255, 255))  # Fill the region of interest
        modified_image = cv2.bitwise_and(image, cv2.bitwise_not(mask))  # Remove the original warped region
        modified_image = cv2.add(image, warped_back)
        cv2.imshow('Modified Image', modified_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        new_four_contours = [transform_contours(contours, M) for contours in four_contours]
        return warped, new_four_contours, M_inv

    def inv_to_original_image(self, image, wrapped_image, inv_transform_matrix, pad_transform_matrix=None):
        if pad_transform_matrix:
            inv_transform_matrix = pad_transform_matrix @ inv_transform_matrix
        # Warp the modified (inpainted) image back to the original perspective
        wrapped_back = cv2.warpPerspective(wrapped_image, inv_transform_matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
        # Create a mask by warping a white image using the same transformation
        white_image = np.ones_like(wrapped_image, dtype=np.uint8) * 255
        mask = cv2.warpPerspective(white_image, inv_transform_matrix, (image.shape[1], image.shape[0]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create a binary mask (adjust threshold as needed)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # Erode the mask slightly to exclude uncertain edge pixels
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Use the mask to blend the original and warped-back images
        modified_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        modified_image = cv2.add(modified_image, cv2.bitwise_and(wrapped_back, wrapped_back, mask=mask))
        
        return modified_image

    def check_valid_dashes_set(self, dashes_set):
        for digit in self.DIGIT_CONTAIN_DASH_INDEX:
            if self.DIGIT_CONTAIN_DASH_INDEX[digit].issuperset(dashes_set):
                return True
        return False

    def compute_digit_from_contours_old(self, contours, image):
        # sort by aspect ratio
        dashes_indexes = set()
        dashes_dict = {}
        max_right = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours])
        
        regular_contours = [contour for contour in contours if compute_contour_aspect_ratio(contour) < 0.5]
        for contour in regular_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # visualize the contour
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.imshow('image', image)   
            # cv2.waitKey(0)
            if w > h: # 5, 6, or 7
                if abs(y-self.all_y_min) <= self.avg_big_y_height / 2:
                    dashes_indexes.add(5)
                    dashes_dict[5] = contour
                elif abs(y-self.all_y_max) <= self.avg_big_y_height / 2:
                    dashes_indexes.add(7)
                    dashes_dict[7] = contour
                else:
                    dashes_indexes.add(6)
                    dashes_dict[6] = contour
            else: # vertical
                if abs(max_right - x) <= self.avg_big_x_width / 2: # on the right
                    if abs(y-self.all_y_min) <= self.avg_big_y_height / 1.2: # should be 2, but perspective transform should work
                        dashes_indexes.add(2)
                        dashes_dict[2] = contour
                    else:
                        dashes_indexes.add(4)
                        dashes_dict[4] = contour
                else:
                    if abs(y-self.all_y_min) <= self.avg_big_y_height / 2:
                        dashes_indexes.add(1)
                        dashes_dict[1] = contour
                    else:
                        dashes_indexes.add(3)
                        dashes_dict[3] = contour                   

        if len(regular_contours) == len(contours):
            # find the set match with DIGIT_CONTAIN_DASH_INDEX
            for digit in self.DIGIT_CONTAIN_DASH_INDEX:
                if self.DIGIT_CONTAIN_DASH_INDEX[digit] == dashes_indexes:
                    return digit, dashes_dict
        
        # second round, deal with fat contours
        bad_contours = [contour for contour in contours if compute_contour_aspect_ratio(contour) >= 0.5]
        min_x = min([cv2.boundingRect(contour)[0] for contour in contours])
        max_right = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours])
        # for now, assume it would not include 6
        for contour in bad_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.imshow('image', image)   
            # cv2.waitKey(0)
            # x max is the thing that we know for sure. Check if this is only on the right
            if abs((x + w) - max_right) >= self.avg_small_x_width / 4: # has to not include anything on the right
                if abs(y-self.all_y_min) <= self.avg_big_y_height / 2:
                    dashes_indexes.add(5)
                    dashes_indexes.add(1)
                else:
                    dashes_indexes.add(7)
                    dashes_indexes.add(3)
            else:
                if abs(y-self.all_y_min) <= self.avg_big_y_height / 1.2:
                    dashes_indexes.add(5)
                    dashes_indexes.add(2)
                    print("adding 2 and 5")
                else:
                    dashes_indexes.add(7)
                    dashes_indexes.add(4)
                    print("adding 4 and 7")


        for digit in self.DIGIT_CONTAIN_DASH_INDEX:
            if self.DIGIT_CONTAIN_DASH_INDEX[digit] == dashes_indexes:
                return digit, dashes_dict
        return -1, dashes_dict
    
    def compute_digit_from_contours(self, contours, image):
        # sort by aspect ratio
        width, height = image.shape[1], image.shape[0]
        dashes_indexes = set()
        dashes_dict = {}
        max_right = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours])
        if self.debug:
            debug_image = image.copy()
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self.debug:
                # visualize the contour
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('digit contours', debug_image)   
                cv2.waitKey(0)
            if w > h: # 5, 6, or 7
                if y < height / 2 - self.avg_big_y_height / 2:
                    dashes_indexes.add(5)
                    dashes_dict[5] = contour
                elif y > height / 2 + self.avg_big_y_height / 2:
                    dashes_indexes.add(7)
                    dashes_dict[7] = contour
                else:
                    dashes_indexes.add(6)
                    dashes_dict[6] = contour
            else: # vertical
                if abs(max_right - x) <= self.avg_big_x_width / 2: # on the right
                    if y <= self.avg_big_y_height / 2:
                        dashes_indexes.add(2)
                        dashes_dict[2] = contour
                    else:
                        dashes_indexes.add(4)
                        dashes_dict[4] = contour
                else:
                    if y <= self.avg_big_y_height / 2:
                        dashes_indexes.add(1)
                        dashes_dict[1] = contour
                    else:
                        dashes_indexes.add(3)
                        dashes_dict[3] = contour                   

        for digit in self.DIGIT_CONTAIN_DASH_INDEX:
            if self.DIGIT_CONTAIN_DASH_INDEX[digit] == dashes_indexes:
                return digit, dashes_dict
        
        return -1, dashes_dict

    def inpaint_with_dark_colors(self, image, mask, brightness_threshold=200):
        """
        Inpaint a region defined by a mask, blending it seamlessly using darker surrounding colors.
        
        Parameters:
            image (np.ndarray): The input image (BGR format).
            mask (np.ndarray): Binary mask where the region to be inpainted is marked (255 in mask).
            brightness_threshold (int): Pixels with brightness above this value will be ignored.
            
        Returns:
            np.ndarray: The inpainted image.
        """
        # Convert the image to a float32 copy for manipulation
        result = image.copy().astype(np.float32)
        mask_indices = np.argwhere(mask == 255)  # Get indices of the region to inpaint

        # Iterate through each pixel in the mask
        for y, x in mask_indices:
            # Sample a 3x3 or 5x5 neighborhood
            neighborhood = result[max(0, y-2):y+3, max(0, x-2):x+3]

            # Compute brightness for all pixels in the neighborhood
            brightness = np.linalg.norm(neighborhood, axis=-1)

            # Filter out bright pixels based on the brightness threshold
            valid_pixels = neighborhood[brightness < brightness_threshold]

            if len(valid_pixels) > 0:
                # Use the mean of valid pixels to fill the current pixel
                result[y, x] = np.mean(valid_pixels, axis=0)
            else:
                # If no valid pixels are found, fallback to black (or another dark color)
                result[y, x] = (0, 0, 0)  # Default fallback color

        # Convert the result back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    
    def hide_dashes(self, image, digit_index, dash_indexes):
        inpainted_image = image.copy()
        
        for dash_index in dash_indexes:
            contour = self.dashes_dicts[digit_index][dash_index]
            contour = np.array(contour, dtype=np.int32)
            mask = np.zeros_like(inpainted_image)

            
            
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # inpainted_image = cv2.inpaint(inpainted_image, gray_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            inpainted_image = self.inpaint_with_dark_colors(inpainted_image, gray_mask, 50)
        inpainted_image = cv2.GaussianBlur(inpainted_image, (3, 3), 0)
        if self.debug:
            cv2.imshow('inpainted_image', inpainted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return inpainted_image 

    def add_dashes(self, image, digit_index, dash_indexes, smart_handle_first_1=True, apply_smoothing=True):
        faked_image = image.copy()
        image_left_offset = 0
        pad_offset_mat = None
        if digit_index == 0 and self.digits[0] == 1 and not smart_handle_first_1: # meaning we have to pad the image whatsoever
            # find the longest width in all dashes of all digits
            image_left_offset = max([cv2.boundingRect(contour)[2] for digit_index in range(1, 4) for contour in self.dashes_dicts[digit_index].values()])
            # pad the image to the left
            faked_image = cv2.copyMakeBorder(faked_image, 0, 0, image_left_offset, 0, cv2.BORDER_CONSTANT, value=0)
            pad_offset_mat = np.float32([[1, 0, image_left_offset], [0, 1, 0]])


        for dash_index in dash_indexes:
            # find a dash to in available_dashes to put in the respective location
            dash_closest_digit_index = sorted(self.available_dashes[dash_index], key=lambda x: abs(x - digit_index))[0]
            contour = self.dashes_dicts[dash_closest_digit_index][dash_index]
            if apply_smoothing:
                epsilon = 0.005 * cv2.arcLength(contour, True)  # Adjust epsilon for smoothness
                contour = cv2.approxPolyDP(contour, epsilon, True)
            # copy to the respective location
            destination_right_most_x = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in self.dashes_dicts[digit_index].values()])
            source_right_most_x = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in self.dashes_dicts[dash_closest_digit_index].values()])
            x_offset = destination_right_most_x - source_right_most_x
            mask = np.zeros_like(faked_image, dtype=np.uint8)
            if apply_smoothing:
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            contour = np.array(contour, dtype=np.int32)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            old_region = cv2.bitwise_and(faked_image, mask)

            if image_left_offset != 0: # mean a left-most one, need to offset these as
                mask = cv2.copyMakeBorder(mask, 0, 0, image_left_offset, 0, cv2.BORDER_CONSTANT, value=0)
                old_region = cv2.copyMakeBorder(old_region, 0, 0, image_left_offset, 0, cv2.BORDER_CONSTANT, value=0)


            translation_matrix = np.float32([[1, 0, x_offset], [0, 1, 0]])
            translated_region = cv2.warpAffine(old_region, translation_matrix, (faked_image.shape[1], faked_image.shape[0]))
            translated_mask = cv2.warpAffine(mask, translation_matrix, (faked_image.shape[1], faked_image.shape[0]))

            mask_inv = cv2.bitwise_not(translated_mask)
            image_without_region = cv2.bitwise_and(faked_image, mask_inv)
            faked_image = cv2.add(image_without_region, translated_region)

        # add a gentle blue
        faked_image = cv2.GaussianBlur(faked_image, (7, 7), 0)
        if self.debug:
            cv2.imshow('faked_image', faked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return faked_image, pad_offset_mat

    def hide_dashes_static(self, image, digit_index, dash_indexes):
        filled_image = image.copy()
        for dash_index in dash_indexes:
            contour = self.dashes_dicts[digit_index][dash_index]
            contour = np.array(contour, dtype=np.int32)
            mask = np.zeros(filled_image.shape[:2], dtype=np.uint8)

            
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contour_coords = np.column_stack(np.where(mask == 255))  # Get current contour area
            for y, x in contour_coords:
            # Define a 3×3 region to the left of the current pixel
                left_region = filled_image[max(0, y - 1):min(y + 2, filled_image.shape[0]),
                                        max(0, x - 3):x]

                if left_region.size > 0:  # Ensure the region exists
                    # Find the darkest pixel in the left region
                    darkest_pixel = np.min(left_region, axis=(0, 1))  # Min intensity for each channel (B, G, R)
                    filled_image[y, x] = darkest_pixel  # Replace current pixel with the darkest pixel
            # kernel = np.ones((3, 3), np.float32) / 9
            # filled_image = cv2.filter2D(filled_image, -1, kernel)
        filled_image = cv2.GaussianBlur(filled_image, (5, 5), 0)
        cv2.imshow('filled_image', filled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return filled_image




    def extract_pixels_from_box(self, box, digit):
        possible_changes = self.find_all_possible_changes(digit)

def compute_contour_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w / h if w <= h else h/w

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

if __name__ == "__main__":
    digit_faker = DigitFaker(digit_detection_method="contour", debug=True)
    # digit_faker.fake_digit('data/clock_only.png')
    digit_faker.fake_digit('data/clock_super_easy.png')
