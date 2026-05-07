package com.suitexen.opencv.processing;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Advanced Noise Reduction Engine
 */
public class ImageDenoiser {

    private final int diameter;
    private final double sigmaColor;
    private final double sigmaSpace;

    public ImageDenoiser() {
        // ★ টিউনড: 7 -> 5, 75 -> 50 (প্রসেসিং ফাস্ট হবে, টেক্সট আরও শার্প থাকবে)
        this(5, 50, 50);
    }

    public ImageDenoiser(int diameter, double sigmaColor, double sigmaSpace) {
        this.diameter = diameter;
        this.sigmaColor = sigmaColor;
        this.sigmaSpace = sigmaSpace;
    }

    public Mat denoise(Mat input) {
        Mat output = new Mat();
        Imgproc.bilateralFilter(input, output, diameter, sigmaColor, sigmaSpace);
        return output;
    }
}