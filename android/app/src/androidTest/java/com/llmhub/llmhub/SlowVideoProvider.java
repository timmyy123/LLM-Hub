package com.llmhub.llmhub;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.os.SystemClock;
import android.provider.OpenableColumns;

import java.io.FileNotFoundException;

/**
 * Test-APK content provider that answers every call after 3 s, like a busy MediaProvider or a
 * cloud-backed picker file. It runs in the test package's own process, so the app's calls cross
 * a binder boundary as they do on users' devices. Java, because that process has no Kotlin runtime.
 */
public class SlowVideoProvider extends ContentProvider {
    public static final String AUTHORITY = "com.llmhub.llmhub.test.slowvideo";
    private static final long DELAY_MS = 3_000L;

    @Override public boolean onCreate() { return true; }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder) {
        SystemClock.sleep(DELAY_MS);
        MatrixCursor cursor = new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME});
        cursor.addRow(new Object[] {uri.getLastPathSegment()});
        return cursor;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        SystemClock.sleep(DELAY_MS);
        throw new FileNotFoundException("slow test video " + uri);
    }

    @Override public String getType(Uri uri) { return "video/mp4"; }
    @Override public Uri insert(Uri uri, ContentValues values) { return null; }
    @Override public int delete(Uri uri, String selection, String[] selectionArgs) { return 0; }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) { return 0; }
}
