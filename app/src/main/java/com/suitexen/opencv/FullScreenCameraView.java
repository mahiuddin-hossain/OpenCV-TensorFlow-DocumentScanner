package com.suitexen.opencv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.util.AttributeSet;

import org.opencv.android.JavaCamera2View;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class FullScreenCameraView extends JavaCamera2View {

    private CvCameraViewListener2 mOwnListener;
    private Bitmap mOwnBitmap;

    // ★ Standard 4:5 ratio (width:height)
    private static final float TARGET_RATIO = 4f / 5f;

    // Overlay & border colors
    private static final int OVERLAY_COLOR = 0xCC000000;   // 80% black
    private static final int BORDER_COLOR  = 0x40FFFFFF;   // 25% white
    private static final float BORDER_WIDTH = 1.5f;

    private final Paint mBorderPaint;

    public FullScreenCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mBorderPaint = new Paint();
        mBorderPaint.setColor(BORDER_COLOR);
        mBorderPaint.setStyle(Paint.Style.STROKE);
        mBorderPaint.setStrokeWidth(BORDER_WIDTH);
        mBorderPaint.setAntiAlias(true);
    }

    @Override
    public void setCvCameraViewListener(CvCameraViewListener2 listener) {
        super.setCvCameraViewListener(listener);
        mOwnListener = listener;
    }

    // ★ No onMeasure override — let it be full screen
    // We handle the 4:5 region inside deliverAndDrawFrame

    @Override
    protected void deliverAndDrawFrame(CvCameraViewFrame frame) {
        Mat modified;
        if (mOwnListener != null) {
            modified = mOwnListener.onCameraFrame(frame);
        } else {
            modified = frame.rgba();
        }

        if (modified == null) return;

        // Create/recreate bitmap if size changed
        if (mOwnBitmap == null
                || mOwnBitmap.getWidth() != modified.width()
                || mOwnBitmap.getHeight() != modified.height()) {
            if (mOwnBitmap != null) mOwnBitmap.recycle();
            mOwnBitmap = Bitmap.createBitmap(
                    modified.width(), modified.height(),
                    Bitmap.Config.ARGB_8888);
        }

        try {
            Utils.matToBitmap(modified, mOwnBitmap);
        } catch (Exception e) {
            return;
        }

        Canvas canvas = getHolder().lockCanvas();
        if (canvas != null) {
            int canvasW = canvas.getWidth();
            int canvasH = canvas.getHeight();
            int bmpW = mOwnBitmap.getWidth();
            int bmpH = mOwnBitmap.getHeight();

            // ★ STEP 1: Calculate 4:5 target rect, centered on canvas
            float targetW, targetH;
            if ((float) canvasW / canvasH > TARGET_RATIO) {
                // Canvas is wider than 4:5 → constrain by height
                targetH = canvasH;
                targetW = canvasH * TARGET_RATIO;
            } else {
                // Canvas is taller than 4:5 → constrain by width
                targetW = canvasW;
                targetH = canvasW / TARGET_RATIO;
            }

            float targetLeft = (canvasW - targetW) / 2f;
            float targetTop  = (canvasH - targetH) / 2f;
            RectF targetRect = new RectF(
                    targetLeft, targetTop,
                    targetLeft + targetW, targetTop + targetH
            );

            // ★ STEP 2: Fill entire canvas with dark overlay
            canvas.drawColor(OVERLAY_COLOR, PorterDuff.Mode.SRC_OVER);

            // ★ STEP 3: Center-crop camera bitmap into 4:5 region
            float scaleX = targetW / bmpW;
            float scaleY = targetH / bmpH;
            float scale = Math.max(scaleX, scaleY);

            float scaledW = bmpW * scale;
            float scaledH = bmpH * scale;

            float bmpLeft = (targetW - scaledW) / 2f + targetLeft;
            float bmpTop  = (targetH - scaledH) / 2f + targetTop;

            // Clip to 4:5 region so nothing spills out
            canvas.save();
            canvas.clipRect(targetRect);
            canvas.drawBitmap(mOwnBitmap, null,
                    new RectF(bmpLeft, bmpTop, bmpLeft + scaledW, bmpTop + scaledH),
                    null);
            canvas.restore();

            // ★ STEP 4: Draw subtle border around 4:5 region
            canvas.drawRect(targetRect, mBorderPaint);

            getHolder().unlockCanvasAndPost(canvas);
        }
    }

    @Override
    public void disableView() {
        super.disableView();
        if (mOwnBitmap != null) {
            mOwnBitmap.recycle();
            mOwnBitmap = null;
        }
    }
}