package com.suitexen.opencv.processing;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class LightingCorrector {

    private final double claheClipLimit;
    private final Size claheGridSize;
    private final boolean autoGamma;

    public LightingCorrector() {
        // ★ টিউনড: 4.0 থেকে 3.0 করা হয়েছে যাতে নয়েজ বেশি না বাড়ে
        this(3.0, new Size(8, 8), true);
    }

    public LightingCorrector(double claheClipLimit, Size claheGridSize, boolean autoGamma) {
        this.claheClipLimit = claheClipLimit;
        this.claheGridSize = claheGridSize;
        this.autoGamma = autoGamma;
    }

    public Mat correct(Mat bgrInput) {
        Mat lab = new Mat();
        Imgproc.cvtColor(bgrInput, lab, Imgproc.COLOR_BGR2Lab);

        List<Mat> labPlanes = new ArrayList<>();
        Core.split(lab, labPlanes);

        CLAHE clahe = Imgproc.createCLAHE(claheClipLimit, claheGridSize);
        clahe.apply(labPlanes.get(0), labPlanes.get(0));

        if (autoGamma) {
            applyAutoGamma(labPlanes.get(0));
        }

        Core.merge(labPlanes, lab);
        Mat result = new Mat();
        Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR);

        lab.release();
        for (Mat m : labPlanes) m.release();

        return result;
    }

    private void applyAutoGamma(Mat lChannel) {
        double mean = Core.mean(lChannel).val[0];
        double gamma = Math.log(128.0 / 255.0) / Math.log(mean / 255.0);
        gamma = Math.max(0.6, Math.min(gamma, 2.0));

        Mat normalized = new Mat();
        lChannel.convertTo(normalized, CvType.CV_64F, 1.0 / 255.0);
        Core.pow(normalized, 1.0 / gamma, normalized);
        normalized.convertTo(lChannel, CvType.CV_8U, 255.0);
        normalized.release();
    }
}