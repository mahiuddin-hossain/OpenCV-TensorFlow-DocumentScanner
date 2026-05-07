package com.suitexen.opencv.processing;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageUpscaler {
    private final double scaleFactor;

    public ImageUpscaler(double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    public Mat upscale(Mat input) {
        Mat output = new Mat();
        Imgproc.resize(
                input,
                output,
                new Size(0, 0),
                scaleFactor,
                scaleFactor,
                Imgproc.INTER_LANCZOS4
        );
        return output;
    }
}