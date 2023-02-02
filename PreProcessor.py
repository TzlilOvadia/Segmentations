from unittest import result
import cv2
import numpy as np
import skimage
from numpy import uint16
from skimage.measure import label
from skimage.morphology import erosion, dilation, closing, opening
import nibabel as nib
from matplotlib import pyplot as plt
import multiprocessing
from skimage.segmentation import morphological_chan_vese
from skimage.transform import hough_circle, hough_circle_peaks
from metrics.scores import dice_coeff, vod_score

"""
#########################################################
################### PROJECT CONSTANTS ###################
#########################################################
"""
MAX_VOXEL_VALUE = 2**16-1
MIN_VOXEL_VALUE = 0
CONNECTED_COMPONENTS = 1
INTENSITY_HISTOGRAM = 2
RAW_DENSITY_HISTOGRAM = 3
WITH_BUFFER_MARGIN = 1.5
PREDICTED_DENSITY_HISTOGRAM = 4
RADIUS_PADDING = 2
WHITE_COLOR = (255, 255, 255)
FILL_SHAPE = -1
HIGHEST_DICE = -1
INITIAL_RADIUS = 17
ROWS_PADDING = 20
COLUMNS_PADDING = 20
GET_X_CENTER, GET_Y_CENTER, GET_CIRCLE_RADIUS = 0,1,2


class PreProcessor:
    def __init__(self, file_path_scan: str, file_path_l1: str, output_directory: str, resolution=14, Imax=1300):
        self.resolution = resolution
        self.filepath = file_path_scan
        self.raw_ct = nib.as_closest_canonical(nib.load(file_path_scan))
        self.output_directory = output_directory
        self.ct_data = self.raw_ct.get_fdata()
        self.raw_l1 = nib.as_closest_canonical(nib.load(file_path_l1))
        self.l1_data = self.raw_l1.get_fdata()
        _, _, z_slices = np.nonzero(self.ct_data)
        self.bottom_bound_slice, self.top_bound_slice = np.min(z_slices), np.max(z_slices)
        self.Imax = Imax

    @staticmethod
    def SegmentationByTH(Imin, Imax, img_data):
        """
        This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
        The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
        :param nifty_file:
        :param Imin:
        :param Imax:
        :return:
        """

        img = np.copy(img_data.astype(dtype=np.uint16))
        img[(img <= Imax) & (img > Imin)] = MAX_VOXEL_VALUE
        img[img < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
        closed_img = skimage.morphology.closing(img)
        return closed_img

    def SkeletonTHFinder(self):
        """
        This function will find the best threshold for the skeleton segmentation, by iterating over Haunsfield units
        in the range of lowerbound (start parameter to stop parameter) to Imax given in the class constructor.
        We use start=150, stop=514 and Imax=1300 as a preset for the sake of the requirements for this assignment
        :return: File containing the skeleton layer, as a result of the threshold application.
        """
        with multiprocessing.Manager() as manager:
            # Prepare processes for task:
            num_cores = multiprocessing.cpu_count()git
            # Prepare tasks distribution between all processes:
            ranges = PreProcessor._tasks_dispatcher(num_cores, start=150, stop=514, resolution=self.resolution)
            # Save process's results in a dictionary, where keys are PIDs and values are [connected components,
            # [threshold images]] v
            results = manager.dict()
            # Create all the processes, according to the number of cores available:
            processes = [multiprocessing.Process(target=PreProcessor.do_segmentation,
                                                 args=(
                                                 ranges[pid], self.Imax, results, pid, self.ct_data)) for
                         pid
                         in range(num_cores)]
            # Execution:
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            cmps = []
            img_threshold_result = []
            for p in range(num_cores):
                pid_ccmps, pid_imgs = results[p]
                cmps.extend(pid_ccmps)
                img_threshold_result.extend(pid_imgs)

            plt.plot(cmps)
            plt.title(f"on file {self.filepath}")
            plt.show()
            # Find all local minima
            dips =  self.find_all_minima(cmps)
            self.bones = img_threshold_result[dips[0]]
            manager.shutdown()
            final_skeleton = nib.Nifti1Image(self.extract_largest_component(self.bones), self.raw_ct.affine)
            case_i = self.filepath.split('.')[0].split('/')[1]
            nib.save(final_skeleton, filename=f"{case_i}_Skeleton_result.nii.gz")
            return self.bones

    @staticmethod
    def do_segmentation(Imin_range, Imax, result_keep, pid, img_data):
        """
        This function is being called by each running process, on a different range of threshold values
        of Haunsfield units (HU). Each process will perform a segmentation by threshold between Imax, which is constant
        for the entire run, to the values in Imin_range, where each iteration we keep the thresholded image, and also
        keep the number of all connected components we got after we applied the threshold on the CT scan.
        we keep the results -> (connecteed components count, thresholded image) to a dictionary, where the key represents the
        the process id.
        :param Imin_range: range of lower bounds for the thresholding of the skeleton. (HU)
        :param Imax: Maximum value of the thresholding (HU)
        :param result_keep: keeping results for each process running this function
        :param pid: process ID
        :param img_data: the image data we want to segment the skeleton from
        :return:
        """
        img_res = []
        ccmps = []
        for i_min in Imin_range:
            img = PreProcessor.SegmentationByTH(Imin=i_min, Imax=Imax, img_data=img_data)
            _, cmp = label(img, return_num=True)
            ccmps.append(cmp)
            img_res.append(img)
        process_results = ccmps, img_res
        result_keep[pid] = process_results

    @staticmethod
    def _find_circles(patched_slice, prev_circle=(0, 0, 0)):
        """
        This function finds the best circles in a given patched CT-scan slice. it does so by finding circles on
        different levels of the 2-level image-pyramid. by using the pyramid, we reduce the 'noisy' circles when we
        eliminate those who has no significant pairing between the different layers.
        :param patched_slice: an approximated ROI of the aorta in each CT-scan slice.
        :param prev_circle: a previously detected circle in a previous slice (if we're not in the first slice)
        :return: best_circle: the best circle in terms of pyramid matched pair, and also similarity circles to the
        prev_circle.
        """
        pyramid = [np.copy(patched_slice)]
        pyramid.append(cv2.pyrDown(np.copy(patched_slice)))
        # Define the parameters for the HoughCircles function
        circles_all_floors = []
        best_circle = None
        for factor, level in enumerate(pyramid):
            if not prev_circle[2]:
                hough_radii = np.arange(10, 18)
            else:
                hough_radii = np.arange(max(7, prev_circle[GET_CIRCLE_RADIUS] - 2),
                                        min(prev_circle[GET_CIRCLE_RADIUS] + 2, 18))
            # Apply HoughCircles to detect circles
            hough_res = hough_circle(level, hough_radii)
            # Select the most prominent 3 circles
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=3)
            cur = np.array([cx, cy, radii]).T
            circles_all_floors.append(cur)
            if factor > 0:
                lower_lev = circles_all_floors[factor - 1]
                best_circles = PreProcessor.find_top_matching_circles(cur, lower_lev,
                                                                      mask_shape=pyramid[factor - 1].shape)
                if not prev_circle[GET_CIRCLE_RADIUS]:
                    return best_circles[0]

                best_circle = PreProcessor.best_circ(best_circles, slice_ROI=patched_slice, prev_circle=prev_circle)
        return best_circle

    @staticmethod
    def best_circ(cur_circles, slice_ROI, prev_circle):
        """

        :param cur_circles:
        :param slice_ROI:
        :param prev_circle:
        :return:
        """
        dices, dist_scores, var_values = [], [], []
        prev_x, prev_y, prev_rad = prev_circle
        prev_mask = np.zeros(slice_ROI.shape).astype(uint16)
        cv2.circle(prev_mask, (int(prev_x), int(prev_y)), int(prev_rad), WHITE_COLOR, FILL_SHAPE)
        for cir in cur_circles:
            x, y, rad = cir[GET_X_CENTER], cir[GET_Y_CENTER], cir[GET_CIRCLE_RADIUS]
            cur_mask = np.zeros(slice_ROI.shape).astype(uint16)
            cv2.circle(cur_mask, (int(x), int(y)), int(rad),WHITE_COLOR, FILL_SHAPE)
            dices.append(dice_coeff(prev_mask, cur_mask))
        dist_score_ord = np.argsort(dices)

        return cur_circles[dist_score_ord[HIGHEST_DICE]]

    @staticmethod
    def find_top_matching_circles(cur, lower_lev, mask_shape: tuple[int, int]):
        """
        Given 2 lists of circles from 2-consecutive slices, this function will find the 3-best circles from the
        previous slice that has the best fitting circle in the current slice, w.r.t dice coefficient.
        :param cur: list of circles from current slice
        :param lower_lev: list of circles from previous slice
        :param mask_shape: the shape of the mask, usually it is the shape of the slice
        :return: a list of length 3 (at most), where each item is a lst (prev_idx, cur_idx, dice) where
        prev_idx, cur_idx, are the indices of the best matching circles between the two consecutive slice
        lists
        """
        all_dice_scores = []
        for idx, circ in enumerate(lower_lev):
            x_prev, y_prev, r_prev = circ
            score = [idx, 0, 0]
            prev_circle_mask = np.zeros(mask_shape)
            cv2.circle(prev_circle_mask, (int(x_prev), int(y_prev)), int(r_prev), WHITE_COLOR, FILL_SHAPE)
            for jdx, cur_circ in enumerate(cur):
                x_cur, y_cur, r_cur = cur_circ
                cur_circle_mask = np.zeros(mask_shape)
                cv2.circle(cur_circle_mask, (int(x_cur * 2), int(y_cur * 2)), int(r_cur * 2), WHITE_COLOR, FILL_SHAPE)
                cur_score = dice_coeff(prev_circle_mask, cur_circle_mask)
                if cur_score > score[2]:
                    score[1] = jdx
                    score[2] = cur_score

            all_dice_scores.append(score)
        all_dice_scores.sort(key=lambda tup: tup[2])
        matched_circles = [lower_lev[i] for i, _, _ in all_dice_scores[-1:-min(3, len(all_dice_scores)):-1]]
        return matched_circles

    @staticmethod
    def _tasks_dispatcher(num_cores, resolution, start=150, stop=514):
        """
        Given a number of processes, we use this dispatcher in order to divide the tasks uniformly across the
        workers.
        :param num_cores: number of cpus in the working computer
        :return: Ranges of H.U values to be  calculated by each process
        """
        jobs =(stop - start) // resolution
        d = jobs // num_cores
        ranges = [
            np.arange(start=start + r * d * resolution, stop=start + r * d * resolution + d * resolution,
                      step=resolution) for r in range(num_cores-1)]
        if not jobs % num_cores:
            ranges.append(np.arange(start=start + (num_cores-1) * d * resolution, stop=start + num_cores * d * resolution,
                      step=resolution))
            return ranges

        ranges.append(np.arange(start=start + (num_cores - 1) * d * resolution, stop=start + num_cores * d * resolution
                                                                                     + (jobs % num_cores) * resolution,
                                step=resolution))
        return ranges


    @staticmethod
    def THFinder(roi_patch):
        """
        Given the appropriate path of where we predicted (or guessed at the first iteration) the aorta location,
        we than calculate the appropriate values of the next threshold (High and low)to apply, in order to extract the correct
        values belong to the Aorta's
        :param roi_patch: Patch from the full slice, where we'll later try to find the aorta using HoughTransform.
        :return:
        """
        roi_indices = np.where((roi_patch > 50) & (roi_patch < 250))
        avg = np.mean(roi_patch[roi_indices])
        std = np.std(roi_patch[roi_indices])
        std = max(int(std), 25)
        return int(avg - std), int(avg + std)

    @staticmethod
    def process_img(img_data):
        """
        Simply processing the image, by applying the histogram equalization and bilateral filtering to reduce
        the noise in each slice, making it easier to generalize the thresholding application stage, which is done right
        after this stage.
        :param start:
        :return:
        """
        img_data[img_data > 250] = 0
        img_data[img_data < 0] = 0
        img_data = img_data.astype(np.uint8)
        for i in range(img_data.shape[2]):
            slice = img_data[:, :, i]
            slice = cv2.equalizeHist(slice)
            bilateral = cv2.bilateralFilter(slice, d=9, sigmaColor=95, sigmaSpace=50)
            img_data[:, :, i] = bilateral.astype(np.uint8)
        return img_data


    def AortaTHFinder(self, gt):
        """
        When Called, this function will will find and segment the Aorta's ROI by using the provided L1 CT-scan to get
        a minimal ROI, by making the assumption that the Aorta should be near the L1 vertebrate.
        This function looks for significant circles in a "close neighborhood" around the aorta, and after such circle
        was found, we apply fine tuning using Chan-Vese to get more accurate segmentation results
        :return: Creating a file named <case_i>_Aorta_segmentation.nii.gz in the root directory of the project.
        This file contains the Aorta's segmentation in the CT scan provided by the user.
        """
        rows_center_start, rows_center_stop, column_center_start, axial_upper_bound, axial_lower_bound, \
        col_border = PreProcessor.find_L1_borders(self.l1_data)
        circ = (0, 0, 0)
        rad = INITIAL_RADIUS
        # Perform image processing on the entire CT-scan before moving on to the segmentation task
        self.ct_data = PreProcessor.process_img(self.ct_data)
        processed_patch = self.ct_data[rows_center_start:rows_center_stop,
                          int(col_border[axial_upper_bound - 1]):int(col_border[axial_upper_bound - 1] + 2 * rad),
                          axial_upper_bound - 1]
        dims = self.ct_data.shape[:2]
        aorta_segmentation = np.ones(self.ct_data.shape)
        aorta_segmentation[:, :, :axial_lower_bound] = 0
        aorta_segmentation[:, :, axial_upper_bound:] = 0
        seg = np.array(processed_patch).astype(np.uint8)

        for axial_idx in range(axial_upper_bound - 1, axial_lower_bound - 1, -1):
            # Recalculate the border of L1 vertebrate:
            border_col = int(max(column_center_start, col_border[axial_idx]))

            # Approximate the correct threshold values according to the recent segmentation ROI:
            threshold_low, threshold_high = PreProcessor.THFinder(seg)
            roi_patch = np.copy(self.ct_data[rows_center_start:rows_center_stop,
                                int(col_border[axial_idx]): int(col_border[axial_idx]) + rad * 2, axial_idx])
            roi_patch[(roi_patch < threshold_low) | (roi_patch > threshold_high)] = 0
            roi_patch[roi_patch > 0] = 1
            roi_patch = dilation(roi_patch)
            roi_patch = PreProcessor.extract_largest_component(roi_patch)

            # Find best circle among all circles using HoughTransform
            res_circ = PreProcessor._find_circles(roi_patch, circ)
            x, y, r = res_circ[GET_X_CENTER], res_circ[GET_Y_CENTER], res_circ[GET_CIRCLE_RADIUS]
            circ = res_circ
            aorta_prediction = np.zeros(roi_patch.shape).astype(np.uint8)

            cv2.circle(aorta_prediction, (x, y), r + RADIUS_PADDING, WHITE_COLOR, FILL_SHAPE)
            # Perform final segmentation using chan-vese algorithm, localized to the approximate circular ROI
            aorta_prediction = morphological_chan_vese(aorta_prediction * roi_patch, num_iter=100, init_level_set='disk',
                                                       smoothing=4)

            # Prepare mask for performing the slicing on the
            aorta_mask = np.zeros(dims)
            aorta_mask[rows_center_start:rows_center_stop, border_col: border_col + rad * 2] = np.copy(aorta_prediction)
            aorta_prediction[aorta_prediction > 0] = 1
            patch = self.ct_data[rows_center_start:rows_center_stop, border_col: border_col + rad * 2, axial_idx - 1]
            seg = np.array(patch * aorta_prediction)

            # Performing the aorta segmentation
            aorta_segmentation[:, :, axial_idx] *= aorta_mask

            # Update the new bounding box for the next slice to segment
            rad = int(r * WITH_BUFFER_MARGIN)
            column_center_start = x + col_border[axial_idx] - r
            rows_center_start += y
            rows_center_stop = rows_center_start + rad
            column_center_start -= rad
            rows_center_start -= rad
            col_border[axial_idx - 1] = x + col_border[axial_idx] - r

            aorta_segmentation[:, :, axial_idx] *= aorta_mask
        dice_score = dice_coeff(gt[:, :, axial_lower_bound: axial_upper_bound],
                                aorta_segmentation[:, :, axial_lower_bound:
                                                         axial_upper_bound])
        case_i = self.filepath.split('.')[0].split('/')[1]
        print(f"case: {case_i}, dice_score is: {dice_score}")
        final = nib.as_closest_canonical(nib.Nifti1Image(aorta_segmentation, self.raw_ct.affine))
        nib.save(final, f"{case_i}_Aorta_segmentation.nii.gz")
        return result

    @staticmethod
    def extract_largest_component(threshold_img, n=1):
        """
        This function should be called after we performed a thresholding for the skeleton.
        It will utilize the result kept in self.bones, and will return the largest connected component, i.e., the
        patience skeleton.
        :param output_directory:
        :return:
        """
        labels = label(threshold_img)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + n
        largestCC_img = threshold_img * largestCC
        largestCC_img = opening(largestCC_img)
        largestCC_img = closing(largestCC_img)
        return largestCC_img

    @staticmethod
    def find_L1_borders(l1_img):
        """
        :param l1_img:
        :return:
        """
        x_nonzero_in, y_nonzero_in, z_nonzero_in = np.nonzero(l1_img)
        axial_upper_bound, axial_lower_bound = np.max(z_nonzero_in), np.min(z_nonzero_in)
        column_center_start = np.max(y_nonzero_in)
        rows_center_start = np.min(x_nonzero_in)
        rows_center_stop = (np.max(x_nonzero_in) + np.min(x_nonzero_in)) // 2
        l1_cpy = np.copy(l1_img)
        mid_col = int(np.min(y_nonzero_in) + (column_center_start - np.min(y_nonzero_in)) // 1.2)
        l1_cpy[:, :mid_col, :] = 0
        col_start_line = np.zeros((l1_img.shape[2],))
        for ax_i in range(axial_upper_bound - 1, axial_lower_bound - 1, -1):
            _, sl = np.where(l1_cpy[:, :, ax_i] > 0)
            try:
                sl = np.max(sl)
                col_start_line[ax_i] = sl
            except:
                col_start_line[ax_i] = int(col_start_line[ax_i + 1])
        holes = np.nonzero(col_start_line)
        upper_holes = np.max(holes)
        lower_holes = np.min(holes)
        col_start_line[upper_holes:axial_upper_bound] = col_start_line[upper_holes]
        col_start_line[axial_lower_bound:lower_holes] = col_start_line[lower_holes]
        return rows_center_start + ROWS_PADDING, rows_center_stop, column_center_start - COLUMNS_PADDING, \
               axial_upper_bound, axial_lower_bound, col_start_line.astype(np.uint)

    @staticmethod
    def find_all_minima(connectivity_cmps):
        """
        Given an array of integers, this function will find all the minima points, and save the indices of all of them
        in the _dips array.
        :return:
        """
        minimas = np.array(connectivity_cmps)
        # Finds all local minima
        return np.where((minimas[1:-1] < minimas[0:-2]) * (
                minimas[1:-1] < minimas[2:]))[0]


if __name__ == "__main__":
    for case_i in range(1, 5):
        s = f"resources/Case{case_i}_CT.nii.gz"
        case = s.split('.')[0].split('/')[1]
        aorta = f"resources/Case{case_i}_Aorta.nii.gz"
        aorta_raw = nib.as_closest_canonical(nib.load(aorta))
        aorta_gt = aorta_raw.get_fdata()
        pp = PreProcessor(file_path_scan=f"resources/Case{case_i}_CT.nii.gz",
                          file_path_l1=f"resources/Case{case_i}_L1.nii.gz", output_directory="")
        pp.SkeletonTHFinder()
        aorta_pred = pp.AortaTHFinder(aorta_gt)
        dice = dice_coeff(aorta_gt, aorta_pred)
        vod = vod_score(aorta_gt, aorta_pred)
        print(f"Result for {case_i} Aorta segmentation:\nDice coefficient {dice} \nVOD score {vod}")
