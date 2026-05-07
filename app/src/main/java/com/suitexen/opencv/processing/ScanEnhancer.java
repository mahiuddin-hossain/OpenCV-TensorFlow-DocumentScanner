package com.suitexen.opencv.processing;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Master Enhancement Pipeline
 *
 * Pipeline Flow:
 *   Raw → Smart Upscale → Denoise → Sharpen → Fix Lighting → Clean Background
 */
public class ScanEnhancer {

    private final ImageUpscaler upscaler;
    private final ImageDenoiser denoiser;
    private final ImageSharpener sharpener;
    private final LightingCorrector lightCorrector;
    private final BackgroundCleaner bgCleaner;

    private static final int MAX_DIMENSION_BEFORE_UPSCALE = 1500;

    public ScanEnhancer() {
        this.upscaler = new ImageUpscaler(2.0);       // 2x Upscale
        this.denoiser = new ImageDenoiser();
        this.sharpener = new ImageSharpener();
        this.lightCorrector = new LightingCorrector();
        this.bgCleaner = new BackgroundCleaner();
    }

    /**
     * Original — Smart Upscale only (no filtering)
     */
    public Mat enhanceOriginal(Mat rgbaInput) {
        return smartUpscale(rgbaInput);
    }

    /**
     * ✨ Magic Color — Upscale → Denoise → Sharpen → Lighting → Background
     */
    public Mat enhanceMagic(Mat rgbaInput) {
        // Step 0: Smart Upscale
        Mat upscaled = smartUpscale(rgbaInput);

        // Convert RGBA → BGR for processing
        Mat bgr = new Mat();
        Imgproc.cvtColor(upscaled, bgr, Imgproc.COLOR_RGBA2BGR);
        upscaled.release();

        // Step 1: Denoise
        Mat denoised = denoiser.denoise(bgr);
        bgr.release();

        // Step 2: Sharpen
        Mat sharpened = sharpener.sharpen(denoised);
        denoised.release();

        // Step 3: Fix Lighting (CLAHE + Gamma)
        Mat lit = lightCorrector.correct(sharpened);
        sharpened.release();

        // Step 4: Clean Background (white paper + color boost)
        Mat cleaned = bgCleaner.cleanMagic(lit);
        lit.release();

        // Convert BGR → RGBA for display
        Mat result = new Mat();
        Imgproc.cvtColor(cleaned, result, Imgproc.COLOR_BGR2RGBA);
        cleaned.release();

        return result;
    }

    /**
     * B&W — Upscale → Denoise → Sharpen → Adaptive Threshold
     */
    public Mat enhanceBW(Mat rgbaInput) {
        // Step 0: Smart Upscale
        Mat upscaled = smartUpscale(rgbaInput);

        // Convert RGBA → BGR
        Mat bgr = new Mat();
        Imgproc.cvtColor(upscaled, bgr, Imgproc.COLOR_RGBA2BGR);
        upscaled.release();

        // Step 1: Denoise
        Mat denoised = denoiser.denoise(bgr);
        bgr.release();

        // Step 2: Sharpen
        Mat sharpened = sharpener.sharpen(denoised);
        denoised.release();

        // Step 3: B&W with adaptive threshold
        Mat cleaned = bgCleaner.cleanBW(sharpened);
        sharpened.release();

        // Convert BGR → RGBA
        Mat result = new Mat();
        Imgproc.cvtColor(cleaned, result, Imgproc.COLOR_BGR2RGBA);
        cleaned.release();

        return result;
    }

    /**
     * Grayscale — Upscale → Denoise → Sharpen → Lighting
     */
    public Mat enhanceGrayscale(Mat rgbaInput) {
        // Step 0: Smart Upscale
        Mat upscaled = smartUpscale(rgbaInput);

        // Convert RGBA → BGR
        Mat bgr = new Mat();
        Imgproc.cvtColor(upscaled, bgr, Imgproc.COLOR_RGBA2BGR);
        upscaled.release();

        // Step 1: Denoise
        Mat denoised = denoiser.denoise(bgr);
        bgr.release();

        // Step 2: Sharpen
        Mat sharpened = sharpener.sharpen(denoised);
        denoised.release();

        // Step 3: Fix Lighting
        Mat lit = lightCorrector.correct(sharpened);
        sharpened.release();

        // Convert BGR → RGBA
        Mat result = new Mat();
        Imgproc.cvtColor(lit, result, Imgproc.COLOR_BGR2RGBA);
        lit.release();

        return result;
    }

    // ★ স্মার্ট আপস্কেল: ছবি ছোট হলে শুধু তখনই 2x করবে
    private Mat smartUpscale(Mat input) {
        double maxDim = Math.max(input.width(), input.height());
        if (maxDim < MAX_DIMENSION_BEFORE_UPSCALE) {
            return upscaler.upscale(input);
        } else {
            // ছবি আগে থেকেই বড়, আপস্কেলের দরকার নেই, ক্লোন করে দিলেই হলো
            Mat clone = new Mat();
            input.copyTo(clone);
            return clone;
        }
    }
}