package com.suitexen.opencv.processing;

import org.opencv.core.Point;

/**
 * Temporal Smoothing Engine for Document Contour Points
 * ★ Tuned for "Magnetic" effect & Real-time fluidity
 */
public class ContourSmoother {

    // ★ Magnetic Smoothing parameters
    private static final double BASE_ALPHA = 0.20;
    private static final double MIN_ALPHA   = 0.02;   // ★ Extremely stable: 2% new (Zero jitter magnetic lock)
    private static final double MAX_ALPHA   = 0.35;   // Fast movement: 35% new (Responsive tracking)
    private static final double JUMP_THRESHOLD = 50.0; // Pixels — sudden change = reset
    private static final double LOCK_MOVEMENT  = 5.0;  // Pixels — tolerates camera noise
    private static final int    LOCK_FRAMES    = 15;   // Frames stable before lock (Faster lock)

    private Point[] smoothedPoints = null;
    private Point[] velocityPoints = null;
    private int stableFrameCount = 0;
    private boolean locked = false;

    public Point[] smooth(Point[] newPoints) {
        // ===== No detection → reset =====
        if (newPoints == null || newPoints.length != 4) {
            smoothedPoints = null;
            velocityPoints = null;
            stableFrameCount = 0;
            locked = false;
            return null;
        }

        // ===== First detection → initialize =====
        if (smoothedPoints == null) {
            smoothedPoints = clonePoints(newPoints);
            velocityPoints = new Point[4];
            for (int i = 0; i < 4; i++) {
                velocityPoints[i] = new Point(0, 0);
            }
            stableFrameCount = 1;
            locked = false;
            return clonePoints(smoothedPoints);
        }

        // ===== Jump detection → document changed =====
        double maxJump = 0;
        for (int i = 0; i < 4; i++) {
            maxJump = Math.max(maxJump, dist(newPoints[i], smoothedPoints[i]));
        }

        if (maxJump > JUMP_THRESHOLD) {
            smoothedPoints = clonePoints(newPoints);
            velocityPoints = new Point[4];
            for (int i = 0; i < 4; i++) {
                velocityPoints[i] = new Point(0, 0);
            }
            stableFrameCount = 1;
            locked = false;
            return clonePoints(smoothedPoints);
        }

        // ===== Calculate adaptive alpha =====
        double avgMovement = 0;
        for (int i = 0; i < 4; i++) {
            avgMovement += dist(newPoints[i], smoothedPoints[i]);
        }
        avgMovement /= 4.0;

        double alpha;
        if (avgMovement < LOCK_MOVEMENT) {
            // ★ Very stable — MAXIMUM smoothness (Magnetic Lock)
            alpha = MIN_ALPHA;
            stableFrameCount++;
        } else if (avgMovement > 30.0) {
            // ★ Fast movement — responsive
            alpha = MAX_ALPHA;
            stableFrameCount = 1;
        } else {
            // ★ Linear interpolation
            double t = (avgMovement - LOCK_MOVEMENT) / (30.0 - LOCK_MOVEMENT);
            alpha = MIN_ALPHA + t * (MAX_ALPHA - MIN_ALPHA);
            stableFrameCount = 1;
        }

        // ★ Apply EMA with strong velocity tracking
        for (int i = 0; i < 4; i++) {
            double vx = newPoints[i].x - smoothedPoints[i].x;
            double vy = newPoints[i].y - smoothedPoints[i].y;

            // Smooth velocity itself
            velocityPoints[i].x = 0.4 * vx + 0.6 * velocityPoints[i].x;
            velocityPoints[i].y = 0.4 * vy + 0.6 * velocityPoints[i].y;

            // ★ Strong prediction factor (0.8) so it actively chases the moving edge
            double predictX = smoothedPoints[i].x + velocityPoints[i].x * 0.8;
            double predictY = smoothedPoints[i].y + velocityPoints[i].y * 0.8;

            // EMA: smoothed = alpha × new + (1 - alpha) × predicted
            smoothedPoints[i].x = alpha * newPoints[i].x + (1 - alpha) * predictX;
            smoothedPoints[i].y = alpha * newPoints[i].y + (1 - alpha) * predictY;
        }

        // ★ Lock detection
        locked = stableFrameCount >= LOCK_FRAMES;

        return clonePoints(smoothedPoints);
    }

    // ===================== State Queries =====================

    public boolean isLocked() {
        return locked;
    }

    public int getStableFrameCount() {
        return stableFrameCount;
    }

    public boolean hasPoints() {
        return smoothedPoints != null;
    }

    public Point[] getCurrentPoints() {
        return smoothedPoints != null ? clonePoints(smoothedPoints) : null;
    }

    public void reset() {
        smoothedPoints = null;
        velocityPoints = null;
        stableFrameCount = 0;
        locked = false;
    }

    // ===================== Helpers =====================

    private double dist(Point a, Point b) {
        return Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
    }

    private Point[] clonePoints(Point[] pts) {
        Point[] clone = new Point[pts.length];
        for (int i = 0; i < pts.length; i++) {
            clone[i] = new Point(pts[i].x, pts[i].y);
        }
        return clone;
    }
}