package com.suitexen.opencv.processing;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class BackgroundCleaner {

    private final int whiteThreshold;
    private final int saturationCap;
    private final double satBoost;

    public BackgroundCleaner() {
        this(150, 50, 1.1);
    }

    public BackgroundCleaner(int whiteThreshold, int saturationCap, double satBoost) {
        this.whiteThreshold = whiteThreshold;
        this.saturationCap = saturationCap;
        this.satBoost = satBoost;
    }

    public Mat cleanMagic(Mat bgrInput) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrInput, hsv, Imgproc.COLOR_BGR2HSV);

        List<Mat> planes = new ArrayList<>();
        Core.split(hsv, planes);
        Mat h = planes.get(0);
        Mat s = planes.get(1);
        Mat v = planes.get(2);

        Mat lowSatMask = new Mat();
        Core.compare(s, new Scalar(saturationCap), lowSatMask, Core.CMP_LT);

        Mat vBoosted = new Mat();
        Core.add(v, new Scalar(40), vBoosted, lowSatMask);
        Core.min(vBoosted, new Scalar(255), vBoosted);
        vBoosted.copyTo(v, lowSatMask);

        Mat maskSat = new Mat();
        Mat maskVal = new Mat();
        Mat paperMask = new Mat();
        Core.compare(s, new Scalar(saturationCap), maskSat, Core.CMP_LT);
        Core.compare(v, new Scalar(whiteThreshold), maskVal, Core.CMP_GT);
        Core.bitwise_and(maskSat, maskVal, paperMask);

        v.setTo(new Scalar(255), paperMask);
        s.setTo(new Scalar(0), paperMask);

        Mat textMask = new Mat();
        Core.bitwise_not(paperMask, textMask);

        Mat sBoosted = new Mat();
        s.convertTo(sBoosted, CvType.CV_8U, satBoost, 0);
        sBoosted.copyTo(s, textMask);

        Core.min(s, new Mat(s.size(), s.type(), new Scalar(255)), s);

        Core.merge(planes, hsv);
        Mat result = new Mat();
        Imgproc.cvtColor(hsv, result, Imgproc.COLOR_HSV2BGR);

        hsv.release();
        for (Mat m : planes) m.release();
        lowSatMask.release();
        vBoosted.release();
        maskSat.release();
        maskVal.release();
        paperMask.release();
        textMask.release();
        sBoosted.release();

        return result;
    }

    public Mat cleanBW(Mat bgrInput) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgrInput, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(gray, gray, new org.opencv.core.Size(3, 3), 0);

        Mat bw = new Mat();
        Imgproc.adaptiveThreshold(
                gray, bw, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY,
                45,   // ★ ফিক্স: 44 এর বদলে 45 দেওয়া হয়েছে (বেজোড়/Odd হতে হবে)
                6
        );

        Mat result = new Mat();
        Imgproc.cvtColor(bw, result, Imgproc.COLOR_GRAY2BGR);

        gray.release();
        bw.release();

        return result;
    }
}