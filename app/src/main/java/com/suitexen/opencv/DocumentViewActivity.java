package com.suitexen.opencv;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.suitexen.opencv.processing.ScanEnhancer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DocumentViewActivity extends AppCompatActivity {

    private static final String TAG = "DocumentView";
    private static final int FILTER_ORIGINAL  = 0;
    private static final int FILTER_MAGIC     = 1;
    private static final int FILTER_BW        = 2;
    private static final int FILTER_GRAYSCALE = 3;

    private ImageView mImageView;
    private TextView mChipOriginal, mChipMagic, mChipBW, mChipGrayscale;
    private Button mBtnRetake, mBtnSave, mBtnShare;
    private ProgressBar mProgressBar; // ★ লোডিং যোগ করা হয়েছে

    private Mat mOriginalMat;
    private Mat mCurrentMat;
    private Bitmap mCurrentBitmap;
    private int mCurrentFilter = FILTER_ORIGINAL;
    private String mImagePath;

    private ScanEnhancer mScanEnhancer;
    // ★ ব্যাকগ্রাউন্ড থ্রেড তৈরি করা হয়েছে যাতে UI ফ্রিজ না হয় (ANR Fix)
    private final ExecutorService processingExecutor = Executors.newSingleThreadExecutor();

    static {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded in DocumentViewActivity");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_document_view);

        mImageView       = findViewById(R.id.scanned_image);
        mChipOriginal    = findViewById(R.id.filter_original);
        mChipMagic       = findViewById(R.id.filter_magic);
        mChipBW          = findViewById(R.id.filter_bw);
        mChipGrayscale   = findViewById(R.id.filter_grayscale);
        mBtnRetake       = findViewById(R.id.btn_retake);
        mBtnSave         = findViewById(R.id.btn_save);
        mBtnShare        = findViewById(R.id.btn_share);
        mProgressBar     = findViewById(R.id.progress_bar); // ★ এক্সএমএল এ একটি ProgressBar যোগ করতে ভুলবেন না

        mScanEnhancer = new ScanEnhancer();

        mImagePath = getIntent().getStringExtra("image_path");

        if (mImagePath == null || !new File(mImagePath).exists()) {
            Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        Mat bgrMat = Imgcodecs.imread(mImagePath);
        if (bgrMat.empty()) {
            Toast.makeText(this, "Error reading image", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        mOriginalMat = new Mat();
        Imgproc.cvtColor(bgrMat, mOriginalMat, Imgproc.COLOR_BGR2RGBA);
        bgrMat.release();

        setupFilterChips();
        applyFilter(FILTER_ORIGINAL);
        mChipOriginal.setSelected(true);

        mBtnRetake.setOnClickListener(v -> finish());
        mBtnSave.setOnClickListener(v -> saveToGallery());
        mBtnShare.setOnClickListener(v -> shareImage());
    }

    private void setupFilterChips() {
        TextView[] chips = {mChipOriginal, mChipMagic, mChipBW, mChipGrayscale};
        for (int i = 0; i < chips.length; i++) {
            final int filterType = i;
            chips[i].setOnClickListener(v -> {
                for (TextView chip : chips) chip.setSelected(false);
                ((TextView) v).setSelected(true);
                applyFilter(filterType);
            });
        }
    }

    // ★ ফিক্স: সব ভারী কাজ ব্যাকগ্রাউন্ড থ্রেডে চলবে
    private void applyFilter(int filterType) {
        mCurrentFilter = filterType;
        showLoading(true);

        processingExecutor.execute(() -> {
            Mat resultMat;
            switch (filterType) {
                case FILTER_MAGIC:
                    resultMat = mScanEnhancer.enhanceMagic(mOriginalMat);
                    break;
                case FILTER_BW:
                    resultMat = mScanEnhancer.enhanceBW(mOriginalMat);
                    break;
                case FILTER_GRAYSCALE:
                    resultMat = mScanEnhancer.enhanceGrayscale(mOriginalMat);
                    break;
                default:
                    resultMat = mScanEnhancer.enhanceOriginal(mOriginalMat);
                    break;
            }

            // পুরানো ম্যাট রিলিজ
            if (mCurrentMat != null) mCurrentMat.release();
            mCurrentMat = resultMat;

            // বিটম্যাপে কনভার্ট
            final Bitmap bitmap = Bitmap.createBitmap(mCurrentMat.cols(), mCurrentMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mCurrentMat, bitmap);

            // UI আপডেট
            runOnUiThread(() -> {
                if (mCurrentBitmap != null) mCurrentBitmap.recycle();
                mCurrentBitmap = bitmap;
                mImageView.setImageBitmap(mCurrentBitmap);
                showLoading(false);
            });
        });
    }

    // ★ লোডিং স্টেট দেখানো/লুকানো
    private void showLoading(boolean isLoading) {
        mProgressBar.setVisibility(isLoading ? View.VISIBLE : View.GONE);
        mImageView.setAlpha(isLoading ? 0.5f : 1.0f);

        mChipOriginal.setEnabled(!isLoading);
        mChipMagic.setEnabled(!isLoading);
        mChipBW.setEnabled(!isLoading);
        mChipGrayscale.setEnabled(!isLoading);
        mBtnSave.setEnabled(!isLoading);
        mBtnShare.setEnabled(!isLoading);
    }

    private void saveToGallery() {
        if (mCurrentBitmap == null) return;

        try {
            ContentValues values = new ContentValues();
            String filename = "docscan_" + System.currentTimeMillis() + ".png";
            values.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/DocScan");
            }

            Uri uri = getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

            if (uri != null) {
                try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                    mCurrentBitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                }
                Toast.makeText(this, "✓ Document saved!", Toast.LENGTH_SHORT).show();
            } else {
                saveToFallback(mCurrentBitmap);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error saving to gallery", e);
            saveToFallback(mCurrentBitmap);
        }
    }

    private void saveToFallback(Bitmap bitmap) {
        try {
            File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "DocScan");
            if (!dir.exists()) dir.mkdirs();

            File file = new File(dir, "docscan_" + System.currentTimeMillis() + ".png");
            FileOutputStream fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();

            MediaStore.Images.Media.insertImage(getContentResolver(),
                    file.getAbsolutePath(), file.getName(), "Document Scan");

            Toast.makeText(this, "✓ Document saved!", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "Fallback save failed", e);
            Toast.makeText(this, "Error saving document", Toast.LENGTH_SHORT).show();
        }
    }

    private void shareImage() {
        if (mCurrentBitmap == null) return;

        try {
            File shareFile = new File(getCacheDir(), "share_doc.png");
            FileOutputStream fos = new FileOutputStream(shareFile);
            mCurrentBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();

            Uri uri = FileProvider.getUriForFile(this,
                    getPackageName() + ".fileprovider", shareFile);

            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("image/png");
            shareIntent.putExtra(Intent.EXTRA_STREAM, uri);
            shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(shareIntent, "Share document"));

        } catch (Exception e) {
            Log.e(TAG, "Error sharing image", e);
            Toast.makeText(this, "Error sharing", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // ★ থ্রেড শাটডাউন
        processingExecutor.shutdownNow();
        if (mOriginalMat != null) mOriginalMat.release();
        if (mCurrentMat != null) mCurrentMat.release();
        if (mCurrentBitmap != null) mCurrentBitmap.recycle();
    }

    @Override
    public void finish() {
        super.finish();
        overridePendingTransition(0, 0);
    }
}