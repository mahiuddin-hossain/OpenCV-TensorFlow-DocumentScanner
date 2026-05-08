package com.suitexen.opencv.processing;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class DocumentSegmenter {

    private static final String TAG = "DocumentSegmenter";
    private static final int INPUT_SIZE = 257;
    private static final String MODEL_FILE = "DeepLabV3-Plus-MobileNet.tflite";

    private Interpreter tflite;
    private boolean isInitialized = false;
    private volatile boolean isRunning = false;

    // ★ GPU এবং NNAPI ডেলিগেট ভেরিয়েবল
    private GpuDelegate gpuDelegate = null;
    private NnApiDelegate nnApiDelegate = null;

    private float[][][][] inputBuffer;
    private float[][][][] outputBuffer;
    private int outputH, outputW, outputC;

    public DocumentSegmenter(Context context) {
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context, MODEL_FILE);

            // ★ অ্যাডভান্সড অপশন সেটআপ (GPU -> NNAPI -> CPU অটো-ফলব্যাক)
            Interpreter.Options options = new Interpreter.Options();

            try {
                // প্রায়োরিটি ১: GPU ডেলিগেট চেষ্টা (সবচেয়ে ফাস্ট)
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
                Log.i(TAG, "✓ GPU Delegate activated (TensorFlow Lite GPU)");
            } catch (Throwable t1) {
                Log.w(TAG, "GPU not available, trying NNAPI: " + t1.getMessage());
                try {
                    // প্রায়োরিটি ২: NNAPI ডেলিগেট (ফোনের AI চিপ/NPU ব্যবহার করবে)
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    Log.i(TAG, "✓ NNAPI Delegate activated (NPU/AI Chip)");
                } catch (Throwable t2) {
                    // প্রায়োরিটি ৩: CPU মাল্টি-থ্রেডেড (ফলব্যাক)
                    options.setNumThreads(4);
                    Log.w(TAG, "⚠ GPU/NNAPI failed, using Multi-threaded CPU (4 threads)");
                }
            }

            tflite = new Interpreter(modelBuffer, options);

            int[] outShape = tflite.getOutputTensor(0).shape();
            outputH = outShape[1];
            outputW = outShape[2];
            outputC = outShape[3];

            inputBuffer = new float[1][INPUT_SIZE][INPUT_SIZE][3];
            outputBuffer = new float[1][outputH][outputW][outputC];

            isInitialized = true;
            Log.i(TAG, "✓ Model loaded: " + INPUT_SIZE + "×" + INPUT_SIZE);

        } catch (Exception e) {
            Log.e(TAG, "✗ Model load failed — will use OpenCV fallback", e);
            isInitialized = false;
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String path) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(path);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = is.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    public Mat segment(Mat rgbaFrame) {
        if (!isInitialized || tflite == null) return null;
        if (isRunning) return null;
        isRunning = true;

        try {
            int origW = rgbaFrame.width();
            int origH = rgbaFrame.height();

            // ★ Aspect Ratio Maintain
            float scale = (float) INPUT_SIZE / Math.max(origW, origH);
            int newW = Math.round(origW * scale);
            int newH = Math.round(origH * scale);

            Mat resizedMat = new Mat();
            Imgproc.resize(rgbaFrame, resizedMat, new Size(newW, newH));

            Mat paddedMat = new Mat(INPUT_SIZE, INPUT_SIZE, resizedMat.type(), new Scalar(0, 0, 0, 0));
            int padX = (INPUT_SIZE - newW) / 2;
            int padY = (INPUT_SIZE - newH) / 2;
            if (newW > 0 && newH > 0) {
                resizedMat.copyTo(paddedMat.submat(padY, padY + newH, padX, padX + newW));
            }
            resizedMat.release();

            byte[] pixelData = new byte[INPUT_SIZE * INPUT_SIZE * 4];
            paddedMat.get(0, 0, pixelData);
            paddedMat.release();

            int pixelIndex = 0;
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    int r = pixelData[pixelIndex++] & 0xFF;
                    int g = pixelData[pixelIndex++] & 0xFF;
                    int b = pixelData[pixelIndex++] & 0xFF;
                    pixelIndex++; // Alpha skip

                    inputBuffer[0][y][x][0] = r / 127.5f - 1.0f;
                    inputBuffer[0][y][x][1] = g / 127.5f - 1.0f;
                    inputBuffer[0][y][x][2] = b / 127.5f - 1.0f;
                }
            }

            Object[] inputs = {inputBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputBuffer);
            tflite.runForMultipleInputsOutputs(inputs, outputs);

            Mat mask = new Mat(outputH, outputW, CvType.CV_8UC1);

            if (outputC == 1) {
                for (int y = 0; y < outputH; y++) {
                    for (int x = 0; x < outputW; x++) {
                        mask.put(y, x, outputBuffer[0][y][x][0] > 0.5f ? (byte) 255 : (byte) 0);
                    }
                }
            } else {
                for (int y = 0; y < outputH; y++) {
                    for (int x = 0; x < outputW; x++) {
                        int maxClass = 0;
                        float maxVal = outputBuffer[0][y][x][0];
                        for (int c = 1; c < outputC; c++) {
                            if (outputBuffer[0][y][x][c] > maxVal) {
                                maxVal = outputBuffer[0][y][x][c];
                                maxClass = c;
                            }
                        }
                        mask.put(y, x, maxClass > 0 ? (byte) 255 : (byte) 0);
                    }
                }
            }

            // ★ প্যাডিং রিমুভ করা
            float maskScaleX = (float) outputW / INPUT_SIZE;
            float maskScaleY = (float) outputH / INPUT_SIZE;

            int maskPadX = (int) (padX * maskScaleX);
            int maskPadY = (int) (padY * maskScaleY);
            int maskNewW = (int) (newW * maskScaleX);
            int maskNewH = (int) (newH * maskScaleY);

            maskPadX = Math.max(0, Math.min(maskPadX, outputW - 1));
            maskPadY = Math.max(0, Math.min(maskPadY, outputH - 1));
            maskNewW = Math.min(maskNewW, outputW - maskPadX);
            maskNewH = Math.min(maskNewH, outputH - maskPadY);

            if (maskNewW <= 0 || maskNewH <= 0) {
                mask.release();
                return null;
            }

            Mat croppedMask = new Mat(mask, new Rect(maskPadX, maskPadY, maskNewW, maskNewH));

            Mat resizedMask = new Mat();
            Imgproc.resize(croppedMask, resizedMask, new Size(origW, origH), 0, 0, Imgproc.INTER_NEAREST);
            Imgproc.threshold(resizedMask, resizedMask, 128, 255, Imgproc.THRESH_BINARY);

            mask.release();
            croppedMask.release();

            return resizedMask;

        } catch (Exception e) {
            Log.e(TAG, "Inference failed, falling back to OpenCV", e);
            return null;
        } finally {
            isRunning = false;
        }
    }

    public boolean isInitialized() {
        return isInitialized;
    }

    public void release() {
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        isInitialized = false;
    }
}