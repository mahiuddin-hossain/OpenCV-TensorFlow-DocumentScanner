package com.suitexen.opencv;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaActionSound;
import android.os.Bundle;
import android.os.Vibrator;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.ImageButton;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.suitexen.opencv.processing.ContourSmoother;
import com.suitexen.opencv.processing.DocumentSegmenter;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private static final int MIN_OUTPUT_DIMENSION = 1500;
    private static final int ML_FRAME_SKIP = 5;

    private static final Scalar COLOR_DETECTED = new Scalar(255, 200, 0, 255);
    private static final Scalar COLOR_STABLE = new Scalar(0, 255, 0, 255);
    private static final Scalar COLOR_CAPTURING = new Scalar(0, 255, 255, 255);

    private CameraBridgeViewBase mOpenCvCameraView;
    private ImageButton mBtnCapture;
    private TextView mStatusText;

    private boolean mIsAutoCapture = false;
    private boolean mCaptureRequested = false;
    private boolean mIsProcessing = false;

    private Mat mLastGray = null;
    private MediaActionSound mShutterSound;

    private DocumentSegmenter mSegmenter;
    private ContourSmoother mContourSmoother;
    private int mMLFrameCounter = 0;

    static {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully via static block");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(1920, 1080);

        mStatusText = findViewById(R.id.status_text);
        RadioGroup modeGroup = findViewById(R.id.mode_group);
        mBtnCapture = findViewById(R.id.btn_capture);

        mShutterSound = new MediaActionSound();
        mShutterSound.load(MediaActionSound.SHUTTER_CLICK);

        mSegmenter = new DocumentSegmenter(this);
        mContourSmoother = new ContourSmoother();

        if (mSegmenter.isInitialized()) {
            Log.i(TAG, "✓ ML segmentation active (every " + ML_FRAME_SKIP + " frames)");
        } else {
            Log.w(TAG, "⚠ ML not available — using OpenCV fallback");
        }

        modeGroup.setOnCheckedChangeListener((group, checkedId) -> {
            mIsAutoCapture = (checkedId == R.id.radio_auto);
            mContourSmoother.reset();
            Toast.makeText(this, mIsAutoCapture ? "Auto Mode" : "Manual Mode", Toast.LENGTH_SHORT).show();
        });

        mBtnCapture.setOnClickListener(v -> {
            if (!mIsProcessing) mCaptureRequested = true;
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            mOpenCvCameraView.setCameraPermissionGranted();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }

        if (OpenCVLoader.initLocal()) mOpenCvCameraView.enableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mIsProcessing = false;
        mCaptureRequested = false;
        mMLFrameCounter = 0;
        mContourSmoother.reset();
        updateStatus("Scan your document");

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            if (mOpenCvCameraView != null) {
                mOpenCvCameraView.setCameraPermissionGranted();
                mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
                if (OpenCVLoader.initLocal()) mOpenCvCameraView.enableView();
            }
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mShutterSound != null) mShutterSound.release();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
        if (mLastGray != null) {
            mLastGray.release();
            mLastGray = null;
        }
        if (mSegmenter != null) mSegmenter.release();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "Camera started: " + width + "x" + height);
    }

    @Override
    public void onCameraViewStopped() {
        Log.d(TAG, "Camera stopped");
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Mat gray = inputFrame.gray();

        if (mLastGray != null) mLastGray.release();
        mLastGray = gray.clone();

        mMLFrameCounter++;

        MatOfPoint2f docContour = detectDocument(rgba, gray);

        if (docContour != null) {
            Scalar contourColor;
            if (mIsAutoCapture && mContourSmoother.isLocked()) {
                contourColor = COLOR_CAPTURING;
                updateStatus("Capturing!");
            } else if (mIsAutoCapture && mContourSmoother.getStableFrameCount() > 10) {
                contourColor = COLOR_STABLE;
                updateStatus("Hold steady...");
            } else {
                contourColor = COLOR_DETECTED;
                updateStatus("Document detected");
            }

            drawContour(rgba, docContour, contourColor);

            if (mIsAutoCapture && !mCaptureRequested && mContourSmoother.isLocked()) {
                mCaptureRequested = true;
            }
        } else {
            updateStatus("Scan your document");
        }

        if (mCaptureRequested && !mIsProcessing) {
            mCaptureRequested = false;
            mIsProcessing = true;
            captureAndProcess(rgba, docContour);
        }

        return rgba;
    }

    private MatOfPoint2f detectDocument(Mat rgba, Mat gray) {
        MatOfPoint2f rawContour = null;
        boolean ranML = false;

        // ★ Priority 1: ML segmentation
        if (mSegmenter.isInitialized() && mMLFrameCounter % ML_FRAME_SKIP == 0) {
            Mat mask = mSegmenter.segment(rgba);
            if (mask != null) {
                rawContour = extractContourFromMask(mask);
                mask.release();
                ranML = true;
            }
        }

        // ★ Priority 2: OpenCV fallback (শুধুমাত্র যখন ML কিছুই পায়নি বা চলছে)
        // ★ ফিক্স: && !mContourSmoother.hasPoints() সরিয়ে দেওয়া হয়েছে যাতে ফলব্যাক সবসময় কাজ করে
        if (rawContour == null) {
            rawContour = findDocumentContour(gray);
        }

        // ★ Apply temporal smoothing
        if (rawContour != null) {
            Point[] pts = rawContour.toArray();
            Point[] smoothed = mContourSmoother.smooth(pts);
            rawContour.release();
            if (smoothed != null) return new MatOfPoint2f(smoothed);
            return null;
        } else if (ranML) {
            mContourSmoother.smooth(null);
            return null;
        } else {
            Point[] cached = mContourSmoother.getCurrentPoints();
            if (cached != null) return new MatOfPoint2f(cached);
            return null;
        }
    }

    private MatOfPoint2f extractContourFromMask(Mat mask) {
        Mat cleaned = new Mat();

        Mat closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.morphologyEx(mask, cleaned, Imgproc.MORPH_CLOSE, closeKernel);

        Mat openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
        Imgproc.morphologyEx(cleaned, cleaned, Imgproc.MORPH_OPEN, openKernel);

        Imgproc.GaussianBlur(cleaned, cleaned, new Size(5, 5), 0);
        Imgproc.threshold(cleaned, cleaned, 128, 255, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cleaned, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        cleaned.release();
        closeKernel.release();
        openKernel.release();
        hierarchy.release();

        if (contours.isEmpty()) {
            for (MatOfPoint c : contours) c.release();
            return null;
        }

        MatOfPoint largest = null;
        double maxArea = 0;
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area > maxArea) {
                maxArea = area;
                largest = c;
            }
        }

        if (largest == null || maxArea < mask.rows() * mask.cols() * 0.05) {
            for (MatOfPoint c : contours) c.release();
            return null;
        }

        MatOfPoint2f c2f = new MatOfPoint2f(largest.toArray());
        double peri = Imgproc.arcLength(c2f, true);

        MatOfPoint2f result = null;
        double[] epsilons = {0.01, 0.015, 0.02, 0.025, 0.03, 0.04};
        for (double eps : epsilons) {
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(c2f, approx, eps * peri, true);
            if (approx.total() == 4 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                result = approx;
                break;
            }
            approx.release();
        }

        if (result == null) {
            MatOfInt hullIndices = new MatOfInt();
            Imgproc.convexHull(largest, hullIndices);
            List<Point> largestPts = largest.toList();
            List<Point> hullPts = new ArrayList<>();
            for (int index : hullIndices.toList()) hullPts.add(largestPts.get(index));
            MatOfPoint hull = new MatOfPoint();
            hull.fromList(hullPts);
            MatOfPoint2f hull2f = new MatOfPoint2f(hull.toArray());
            double hullPeri = Imgproc.arcLength(hull2f, true);
            for (double eps : epsilons) {
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(hull2f, approx, eps * hullPeri, true);
                if (approx.total() == 4 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                    result = approx;
                    break;
                }
                approx.release();
            }
            hullIndices.release();
            hull.release();
            hull2f.release();
        }

        c2f.release();
        for (MatOfPoint c : contours) c.release();
        return result;
    }

    // ===================== OpenCV Fallback (আগের মতোই একই থাকবে) =====================
    private MatOfPoint2f findDocumentContour(Mat gray) {
        double maxDim = Math.max(gray.width(), gray.height());
        double scale = 640.0 / maxDim;
        Size smallSize = new Size(Math.round(gray.width() * scale), Math.round(gray.height() * scale));
        Mat small = new Mat();
        Imgproc.resize(gray, small, smallSize);

        Mat blurred = new Mat();
        Imgproc.GaussianBlur(small, blurred, new Size(5, 5), 0);

        double median = getMedian(blurred);
        Mat cannyEdges = new Mat();
        Imgproc.Canny(blurred, cannyEdges, Math.max(0, 0.5 * median), Math.min(255, 1.5 * median));

        Mat adaptiveEdges = new Mat();
        Imgproc.adaptiveThreshold(blurred, adaptiveEdges, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 15, 5);

        Mat combined = new Mat();
        Core.bitwise_or(cannyEdges, adaptiveEdges, combined);

        Mat closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(combined, combined, Imgproc.MORPH_CLOSE, closeKernel);
        Mat dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(combined, combined, dilateKernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(combined, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        double frameArea = small.rows() * small.cols();
        double centerX = small.width() / 2.0;
        double centerY = small.height() / 2.0;

        contours.sort((c1, c2) -> Double.compare(Imgproc.contourArea(c2), Imgproc.contourArea(c1)));

        MatOfPoint2f bestContour = null;
        double bestScore = -1;

        int limit = Math.min(contours.size(), 20);
        for (int i = 0; i < limit; i++) {
            MatOfPoint contour = contours.get(i);
            double area = Imgproc.contourArea(contour);
            if (area < frameArea * 0.02) break;
            if (area > frameArea * 0.98) continue;

            MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(c2f, true);
            MatOfPoint2f approx = null;
            double[] epsilons = {0.015, 0.02, 0.03, 0.04, 0.05};
            for (double eps : epsilons) {
                MatOfPoint2f temp = new MatOfPoint2f();
                Imgproc.approxPolyDP(c2f, temp, eps * peri, true);
                if (temp.total() == 4 && Imgproc.isContourConvex(new MatOfPoint(temp.toArray()))) {
                    if (isGoodRectangle(temp)) {
                        approx = temp;
                        break;
                    }
                }
                temp.release();
            }
            c2f.release();

            if (approx != null) {
                double score = scoreContourAdvanced(approx, frameArea, centerX, centerY);
                if (score > bestScore) {
                    if (bestContour != null) bestContour.release();
                    bestContour = approx;
                    bestScore = score;
                } else {
                    approx.release();
                }
            }
        }

        MatOfPoint2f result = null;
        if (bestContour != null) {
            Point[] pts = bestContour.toArray();
            for (int j = 0; j < pts.length; j++) {
                pts[j].x = pts[j].x / scale;
                pts[j].y = pts[j].y / scale;
            }
            result = new MatOfPoint2f(pts);
            bestContour.release();
        }

        small.release();
        blurred.release();
        cannyEdges.release();
        adaptiveEdges.release();
        combined.release();
        closeKernel.release();
        dilateKernel.release();
        hierarchy.release();
        for (MatOfPoint c : contours) c.release();

        return result;
    }

    // ===================== বাকি হেল্পার মেথডগুলো (আগের মতোই একই) =====================
    private double scoreContourAdvanced(MatOfPoint2f contour, double frameArea, double centerX, double centerY) {
        Point[] pts = contour.toArray();
        double area = Imgproc.contourArea(contour);
        double areaRatio = area / frameArea;
        if (areaRatio < 0.02 || areaRatio > 0.95) return 0;
        double areaScore = Math.min(areaRatio / 0.3, 1.0);
        double cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.0;
        double cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4.0;
        double maxDist = Math.sqrt(centerX * centerX + centerY * centerY);
        double dist = Math.sqrt(Math.pow(cx - centerX, 2) + Math.pow(cy - centerY, 2));
        double centerScore = 1.0 - (dist / maxDist);
        double angleScore = 0;
        for (int i = 0; i < 4; i++) {
            angleScore += 1.0 - Math.abs(getAngle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) - 90) / 30.0;
        }
        angleScore /= 4.0;
        double w = distance(pts[0], pts[1]);
        double h = distance(pts[0], pts[3]);
        if (w <= 0 || h <= 0) return 0;
        double ratio = Math.max(w, h) / Math.min(w, h);
        if (ratio > 6.0) return 0;
        double ratioScore = 1.0 - Math.min((ratio - 1.0) / 5.0, 1.0);
        return areaScore * 0.25 + centerScore * 0.40 + angleScore * 0.25 + ratioScore * 0.10;
    }

    private boolean isGoodRectangle(MatOfPoint2f contour) {
        Point[] pts = contour.toArray();
        for (int i = 0; i < 4; i++) {
            double angle = getAngle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]);
            if (angle < 60 || angle > 120) return false;
        }
        return true;
    }

    private double getMedian(Mat gray) {
        int histSize = 256;
        Mat hist = new Mat();
        Imgproc.calcHist(Collections.singletonList(gray), new MatOfInt(0), new Mat(), hist, new MatOfInt(histSize), new MatOfFloat(0, 256));
        long total = gray.rows() * gray.cols();
        long sum = 0;
        float[] histData = new float[histSize];
        hist.get(0, 0, histData);
        double median = 128;
        for (int i = 0; i < histSize; i++) {
            sum += histData[i];
            if (sum >= total / 2) {
                median = i;
                break;
            }
        }
        hist.release();
        return median;
    }

    private double getAngle(Point p1, Point p2, Point p3) {
        double a = distance(p2, p3);
        double b = distance(p1, p3);
        double c = distance(p1, p2);
        if (a * c == 0) return 0;
        double cos = (a * a + c * c - b * b) / (2 * a * c);
        cos = Math.max(-1.0, Math.min(1.0, cos));
        return Math.toDegrees(Math.acos(cos));
    }

    private void drawContour(Mat rgba, MatOfPoint2f contour, Scalar color) {
        Point[] points = contour.toArray();
        Point[] sorted = sortPoints(points);
        for (int i = 0; i < 4; i++) Imgproc.line(rgba, sorted[i], sorted[(i + 1) % 4], color, 4);
        for (Point p : sorted) {
            Imgproc.circle(rgba, p, 12, color, -1);
            Imgproc.circle(rgba, p, 14, new Scalar(255, 255, 255, 255), 2);
        }
    }

    private void captureAndProcess(Mat rgba, MatOfPoint2f docContour) {
        try {
            if (mShutterSound != null) mShutterSound.play(MediaActionSound.SHUTTER_CLICK);
        } catch (Exception e) {
            Log.e(TAG, "Error playing shutter sound", e);
        }
        try {
            Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
            if (vibrator != null && vibrator.hasVibrator()) vibrator.vibrate(80);
        } catch (SecurityException e) {
            Log.w(TAG, "Vibrate permission not granted", e);
        }

        if (docContour != null && mLastGray != null) {
            try {
                Imgproc.cornerSubPix(mLastGray, docContour, new Size(11, 11), new Size(-1, -1), new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.01));
            } catch (Exception e) {
                Log.w(TAG, "Corner refinement failed", e);
            }
        }

        Mat result = (docContour != null) ? warpDocument(rgba, docContour) : rgba.clone();
        Bitmap bitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, bitmap);
        result.release();
        String path = saveTempImage(bitmap);
        bitmap.recycle();

        runOnUiThread(() -> {
            mIsProcessing = false;
            if (path != null) {
                Intent intent = new Intent(MainActivity.this, DocumentViewActivity.class);
                intent.putExtra("image_path", path);
                startActivity(intent);
            } else {
                Toast.makeText(MainActivity.this, "Error saving image", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private String saveTempImage(Bitmap bitmap) {
        try {
            File oldFile = new File(getCacheDir(), "scan_temp.png");
            if (oldFile.exists()) oldFile.delete();
            File file = new File(getCacheDir(), "scan_temp.png");
            FileOutputStream fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();
            Log.d(TAG, "Saved: " + file.getAbsolutePath() + " (" + bitmap.getWidth() + "x" + bitmap.getHeight() + ")");
            return file.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "Error saving temp image", e);
            return null;
        }
    }

    private Mat warpDocument(Mat src, MatOfPoint2f contour) {
        Point[] sorted = sortPoints(contour.toArray());
        double wTop = distance(sorted[0], sorted[1]);
        double wBottom = distance(sorted[3], sorted[2]);
        double rawWidth = Math.max(wTop, wBottom);
        double hLeft = distance(sorted[0], sorted[3]);
        double hRight = distance(sorted[1], sorted[2]);
        double rawHeight = Math.max(hLeft, hRight);
        double scale = 1.0;
        double maxDim = Math.max(rawWidth, rawHeight);
        if (maxDim < MIN_OUTPUT_DIMENSION) scale = (double) MIN_OUTPUT_DIMENSION / maxDim;
        int maxWidth = Math.max((int) Math.round(rawWidth * scale), 10);
        int maxHeight = Math.max((int) Math.round(rawHeight * scale), 10);
        MatOfPoint2f srcPoints = new MatOfPoint2f(sorted);
        MatOfPoint2f dstPoints = new MatOfPoint2f(new Point(0, 0), new Point(maxWidth - 1, 0), new Point(maxWidth - 1, maxHeight - 1), new Point(0, maxHeight - 1));
        Mat transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Mat doc = new Mat(maxHeight, maxWidth, CvType.CV_8UC4);
        Imgproc.warpPerspective(src, doc, transform, new Size(maxWidth, maxHeight), Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
        transform.release();
        srcPoints.release();
        dstPoints.release();
        return doc;
    }

    private Point[] sortPoints(Point[] src) {
        Point[] result = new Point[4];
        List<Point> pts = new ArrayList<>();
        Collections.addAll(pts, src);
        pts.sort((p1, p2) -> Double.compare(p1.x + p1.y, p2.x + p2.y));
        result[0] = pts.get(0);
        result[2] = pts.get(3);
        pts.sort((p1, p2) -> Double.compare(p1.y - p1.x, p2.y - p2.x));
        result[1] = pts.get(0);
        result[3] = pts.get(3);
        return result;
    }

    private double distance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }

    private void updateStatus(String text) {
        if (mStatusText != null) mStatusText.post(() -> mStatusText.setText(text));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setCameraPermissionGranted();
                if (OpenCVLoader.initLocal()) mOpenCvCameraView.enableView();
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}