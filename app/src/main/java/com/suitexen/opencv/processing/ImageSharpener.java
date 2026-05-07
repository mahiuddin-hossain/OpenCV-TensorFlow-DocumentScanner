package com.suitexen.opencv.processing;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Advanced Sharpening Engine (Unsharp Mask)
 */
public class ImageSharpener {

    private final double sigma;
    private final double amount;
    private final double threshold;

    public ImageSharpener() {
        // ★ টিউনড: amount 1.2 -> 1.0 (হ্যালো ইফেক্ট কমানো হয়েছে), threshold 2.0 -> 3.0 (নয়েজ শার্প হবে না)
        this(1.5, 1.0, 3.0);
    }

    public ImageSharpener(double sigma, double amount, double threshold) {
        this.sigma = sigma;
        this.amount = amount;
        this.threshold = threshold;
    }

    public Mat sharpen(Mat input) {
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(input, blurred, new Size(0, 0), sigma);

        Mat sharp = new Mat();
        Core.addWeighted(input, 1.0 + amount, blurred, -amount, 0, sharp);

        // ★ ফিক্স: ইন-প্লেস মডিফিকেশন বন্ধ করা হয়েছে। নতুন ম্যাট রিটার্ন হচ্ছে।
        if (threshold > 0) {
            Mat diff = new Mat();
            Core.absdiff(input, blurred, diff);

            Mat mask = new Mat();
            Imgproc.threshold(diff, mask, threshold, 255, Imgproc.THRESH_BINARY);

            Mat result = new Mat();
            input.copyTo(result); // বেস হিসেবে অরিজিনাল কপি করা হলো
            sharp.copyTo(result, mask); // শুধু এজগুলো শার্প করা হলো

            diff.release();
            mask.release();
            sharp.release();
            blurred.release();

            return result;
        }

        blurred.release();
        return sharp;
    }
}