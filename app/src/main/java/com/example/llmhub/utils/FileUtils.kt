package com.llmhub.llmhub.utils

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import com.llmhub.llmhub.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.IOException
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import java.io.StringReader
import java.util.zip.ZipInputStream
import java.util.zip.ZipEntry
import com.itextpdf.kernel.pdf.PdfDocument
import com.itextpdf.kernel.pdf.PdfReader
import com.itextpdf.kernel.pdf.canvas.parser.PdfTextExtractor

/**
 * Utility class for handling various file types in chat attachments
 */
object FileUtils {
    
    /**
     * Supported file types for attachments
     */
    enum class SupportedFileType(
        val mimeTypes: List<String>,
        val extensions: List<String>,
        val displayName: String,
        val icon: String
    ) {
        IMAGE(
            mimeTypes = listOf("image/*"),
            extensions = listOf("jpg", "jpeg", "png", "gif", "bmp", "webp"),
            displayName = "Image",
            icon = "🖼️"
        ),
        PDF(
            mimeTypes = listOf("application/pdf"),
            extensions = listOf("pdf"),
            displayName = "PDF Document",
            icon = "📄"
        ),
        TEXT(
            mimeTypes = listOf("text/plain", "text/*"),
            extensions = listOf("txt", "md", "csv", "log"),
            displayName = "Text File",
            icon = "📝"
        ),
        WORD(
            mimeTypes = listOf(
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            extensions = listOf("doc", "docx"),
            displayName = "Word Document",
            icon = "📘"
        ),
        EXCEL(
            mimeTypes = listOf(
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            extensions = listOf("xls", "xlsx"),
            displayName = "Excel Spreadsheet",
            icon = "📊"
        ),
        POWERPOINT(
            mimeTypes = listOf(
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ),
            extensions = listOf("ppt", "pptx"),
            displayName = "PowerPoint Presentation",
            icon = "📽️"
        ),
        JSON(
            mimeTypes = listOf("application/json"),
            extensions = listOf("json"),
            displayName = "JSON File",
            icon = "⚙️"
        ),
        XML(
            mimeTypes = listOf("application/xml", "text/xml"),
            extensions = listOf("xml"),
            displayName = "XML File",
            icon = "🔧"
        ),
        AUDIO(
            mimeTypes = listOf("audio/*", "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg"),
            extensions = listOf("wav", "mp3", "ogg", "m4a", "aac", "flac"),
            displayName = "Audio File",
            icon = "🎵"
        ),
        UNKNOWN(
            mimeTypes = listOf(),
            extensions = listOf(),
            displayName = "Unknown File",
            icon = "📎"
        )
    }
    
    /**
     * Get file information from URI
     */
    data class FileInfo(
        val name: String,
        val size: Long,
        val mimeType: String?,
        val type: SupportedFileType,
        val uri: Uri
    )
    
    /**
     * Get all supported MIME types for file picker
     */
    fun getAllSupportedMimeTypes(): Array<String> {
        return SupportedFileType.values().flatMap { it.mimeTypes }.toTypedArray()
    }
    
    /**
     * Get localized display name for file type
     */
    fun getLocalizedDisplayName(context: Context, fileType: SupportedFileType): String {
        return when (fileType) {
            SupportedFileType.TEXT -> context.getString(R.string.text_file)
            SupportedFileType.IMAGE -> context.getString(R.string.images)
            SupportedFileType.PDF -> context.getString(R.string.documents)
            SupportedFileType.WORD -> context.getString(R.string.documents)
            SupportedFileType.EXCEL -> context.getString(R.string.documents)
            SupportedFileType.POWERPOINT -> context.getString(R.string.documents)
            SupportedFileType.JSON -> context.getString(R.string.read_this)
            SupportedFileType.XML -> context.getString(R.string.read_this)
            SupportedFileType.AUDIO -> context.getString(R.string.audio_file)
            else -> fileType.displayName
        }
    }
    
    /**
     * Get file information from URI
     */
    suspend fun getFileInfo(context: Context, uri: Uri): FileInfo? = withContext(Dispatchers.IO) {
        try {
            val contentResolver = context.contentResolver
            var name = "Unknown file"
            var size = 0L
            var mimeType: String? = null
            
            android.util.Log.d("FileUtils", "Getting file info for URI: $uri")
            
            // Try to get file info using cursor
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                    
                    if (nameIndex != -1) {
                        val fileName = cursor.getString(nameIndex)
                        name = fileName ?: "Unknown file"
                        android.util.Log.d("FileUtils", "File name from cursor: $fileName")
                    }
                    if (sizeIndex != -1) {
                        val cursorSize = cursor.getLong(sizeIndex)
                        android.util.Log.d("FileUtils", "File size from cursor: $cursorSize")
                        
                        // Check for potential overflow or negative values
                        if (cursorSize < 0) {
                            android.util.Log.w("FileUtils", "Negative file size detected: $cursorSize, will get actual size")
                            size = 0L
                        } else if (cursorSize > 1024L * 1024L * 1024L * 10L) { // More than 10GB
                            android.util.Log.w("FileUtils", "Suspiciously large file size: $cursorSize, will verify actual size")
                            size = cursorSize
                        } else {
                            size = cursorSize
                        }
                    }
                }
            }
            
            // If cursor size seems unreliable or missing, get actual size by reading the stream
            // This is more reliable than OpenableColumns.SIZE which can be incorrect on some devices
            var needsActualSizeCheck = false
            
            if (size <= 0) {
                android.util.Log.d("FileUtils", "Cursor size is 0 or negative, will get actual size")
                needsActualSizeCheck = true
            } else if (size > 1024L * 1024L * 1024L * 10L) { // More than 10GB
                android.util.Log.d("FileUtils", "Cursor size is suspiciously large (${formatFileSize(size)}), will verify")
                needsActualSizeCheck = true
            }
            
            if (needsActualSizeCheck) {
                try {
                    android.util.Log.d("FileUtils", "Getting actual file size by reading input stream...")
                    val actualSize = getActualFileSize(context, uri)
                    if (actualSize > 0) {
                        android.util.Log.d("FileUtils", "Actual file size from stream: $actualSize, cursor size was: $size")
                        
                        // Check if cursor size might be in KB instead of bytes (common issue)
                        val sizeInKB = size * 1024
                        val sizeDiffFromBytes = if (size > 0) Math.abs(actualSize - size).toDouble() / Math.max(actualSize, size) else 1.0
                        val sizeDiffFromKB = if (size > 0) Math.abs(actualSize - sizeInKB).toDouble() / Math.max(actualSize, sizeInKB) else 1.0
                        
                        if (sizeDiffFromKB < 0.1 && sizeDiffFromBytes > 0.5) {
                            // Cursor was likely returning KB instead of bytes
                            android.util.Log.w("FileUtils", "Cursor size ($size) appears to be in KB, actual size is $actualSize bytes")
                            size = actualSize
                        } else if (size <= 0 || sizeDiffFromBytes > 0.3) {
                            // Cursor size was way off, use actual size
                            android.util.Log.w("FileUtils", "Cursor size ($size) was incorrect, using actual size ($actualSize)")
                            size = actualSize
                        } else {
                            // Cursor size seems reasonable, keep it
                            android.util.Log.d("FileUtils", "Cursor size seems accurate, keeping it")
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.w("FileUtils", "Failed to get actual file size: ${e.message}")
                    // If we can't get actual size and cursor size seems wrong, try to make best guess
                    if (size > 1024L * 1024L * 1024L * 10L) {
                        android.util.Log.w("FileUtils", "Using potentially incorrect cursor size: $size")
                    }
                    if (size <= 0) {
                        size = 0L // Reset to 0 if negative
                    }
                }
            }
            
            android.util.Log.d("FileUtils", "Final file size: $size")
            android.util.Log.d("FileUtils", "File size formatted: ${formatFileSize(size)}")
            
            // Get MIME type
            mimeType = contentResolver.getType(uri)
            android.util.Log.d("FileUtils", "MIME type from resolver: $mimeType")
            
            // If MIME type is null, try to infer from extension
            if (mimeType == null) {
                val extension = getFileExtension(name)
                mimeType = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension)
                android.util.Log.d("FileUtils", "MIME type inferred from extension '$extension': $mimeType")
            }
            
            // Determine file type
            val fileType = determineFileType(mimeType, name)
            android.util.Log.d("FileUtils", "Determined file type: ${fileType.displayName}")
            
            val fileInfo = FileInfo(name, size, mimeType, fileType, uri)
            android.util.Log.d("FileUtils", "Created FileInfo: $fileInfo")
            fileInfo
        } catch (e: Exception) {
            android.util.Log.e("FileUtils", "Error getting file info: ${e.message}", e)
            null
        }
    }
    
    /**
     * Get actual file size by reading the input stream
     * This is more reliable than OpenableColumns.SIZE which can be incorrect
     */
    private suspend fun getActualFileSize(context: Context, uri: Uri): Long = withContext(Dispatchers.IO) {
        var totalBytes = 0L
        try {
            context.contentResolver.openInputStream(uri)?.use { inputStream ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var maxBytesToRead = 1024L * 1024L * 1024L * 11L // Don't read more than 11GB to avoid memory issues
                
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    totalBytes += bytesRead
                    
                    // Safety check to avoid memory issues with extremely large files
                    if (totalBytes > maxBytesToRead) {
                        android.util.Log.w("FileUtils", "File is larger than 11GB, stopping size calculation at $totalBytes bytes")
                        break
                    }
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("FileUtils", "Error reading file size from stream: ${e.message}")
            throw e
        }
        totalBytes
    }
    
    /**
     * Determine file type from MIME type or filename
     */
    private fun determineFileType(mimeType: String?, filename: String): SupportedFileType {
        val extension = getFileExtension(filename).lowercase()
        
        return SupportedFileType.values().find { fileType ->
            // Don't match UNKNOWN here, it's a fallback
            if (fileType == SupportedFileType.UNKNOWN) return@find false
            
            // Check by MIME type first
            mimeType?.let { mime ->
                fileType.mimeTypes.any { supportedMime ->
                    if (supportedMime.endsWith("/*")) {
                        mime.startsWith(supportedMime.removeSuffix("/*"))
                    } else {
                        mime.equals(supportedMime, ignoreCase = true)
                    }
                }
            } == true || 
            // Fallback to extension check
            fileType.extensions.contains(extension)
        } ?: SupportedFileType.UNKNOWN  // Return UNKNOWN if no match found
    }
    
    /**
     * Get file extension from filename
     */
    private fun getFileExtension(filename: String): String {
        return filename.substringAfterLast('.', "")
    }
    
    /**
     * Get file name from URI
     */
    private fun getFileName(context: Context, uri: Uri): String? {
        var name: String? = null
        val contentResolver = context.contentResolver
        
        // Try to get filename from content resolver
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex != -1) {
                    name = cursor.getString(nameIndex)
                }
            }
        }
        
        return name
    }
    
    /**
     * Extract text content from supported file types
     */
    suspend fun extractTextContent(context: Context, uri: Uri, fileType: SupportedFileType): String? = 
        withContext(Dispatchers.IO) {
            try {
                when (fileType) {
                    SupportedFileType.TEXT, SupportedFileType.JSON, SupportedFileType.XML -> {
                        extractPlainTextContent(context, uri)
                    }
                    SupportedFileType.PDF -> {
                        extractPdfTextContent(context, uri)
                    }
                    SupportedFileType.WORD -> {
                        extractWordTextContent(context, uri)
                    }
                    SupportedFileType.EXCEL -> {
                        extractExcelTextContent(context, uri)
                    }
                    SupportedFileType.POWERPOINT -> {
                        extractPowerPointTextContent(context, uri)
                    }
                    SupportedFileType.IMAGE -> {
                        null // Images are handled separately by vision models
                    }
                    SupportedFileType.AUDIO -> {
                        null // Audio is handled separately by audio models
                    }
                    SupportedFileType.UNKNOWN -> {
                        "[Unknown file type - Content extraction not available]"
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Error extracting text content: ${e.message}")
                "[Error reading file content]"
            }
        }
    
    /**
     * Extract plain text content from text files
     */
    private suspend fun extractPlainTextContent(context: Context, uri: Uri): String? = 
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8)).use { reader ->
                        val content = reader.readText()
                        
                        // Check if this might be a CSV file and format it nicely
                        val isLikelyCSV = content.contains(',') && content.lines().size > 1
                        
                        val processedContent = if (isLikelyCSV) {
                            try {
                                formatCSVContent(content)
                            } catch (e: Exception) {
                                // If CSV parsing fails, return as plain text
                                content
                            }
                        } else {
                            content
                        }
                        
                        // Limit content size to prevent overwhelming the model
                        if (processedContent.length > 10000) {
                            processedContent.take(10000) + "\n\n[Content truncated - showing first 10,000 characters]"
                        } else {
                            processedContent
                        }
                    }
                }
            } catch (e: IOException) {
                android.util.Log.e("FileUtils", "Error reading text file: ${e.message}")
                null
            }
        }
    
    /**
     * Format CSV content for better readability using Apache Commons CSV
     */
    private fun formatCSVContent(content: String): String {
        return try {
            val parser = CSVFormat.DEFAULT.parse(StringReader(content))
            val records = parser.records
            
            if (records.isEmpty()) return content
            
            val result = StringBuilder()
            result.append("## CSV Data\n\n")
            
            // Get headers from first record
            val firstRecord = records.first()
            val headers = (0 until firstRecord.size()).map { "Column ${it + 1}" }
            
            // Format as table
            result.append("| ${headers.joinToString(" | ")} |\n")
            result.append("|${headers.joinToString("") { " --- |" }}\n")
            
            // Add data rows (limit to first 100 rows)
            records.take(100).forEach { record ->
                val values = (0 until record.size()).map { record.get(it) }
                result.append("| ${values.joinToString(" | ")} |\n")
            }
            
            if (records.size > 100) {
                result.append("\n[Showing first 100 rows of ${records.size} total rows]")
            }
            
            parser.close()
            result.toString()
        } catch (e: Exception) {
            // Fallback to simple formatting
            val lines = content.lines().take(50)
            "## CSV Data\n\n${lines.joinToString("\n")}" + 
            if (content.lines().size > 50) "\n\n[Content truncated - showing first 50 lines]" else ""
        }
    }
    
    /**
     * Extract text content from PDF files using iText7
     */
    private suspend fun extractPdfTextContent(context: Context, uri: Uri): String? = 
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    val pdfReader = PdfReader(inputStream)
                    val pdfDocument = PdfDocument(pdfReader)
                    
                    val text = StringBuilder()
                    val numberOfPages = pdfDocument.numberOfPages
                    
                    // Extract text from all pages (limit to first 20 pages to avoid overwhelming)
                    val pagesToProcess = minOf(numberOfPages, 20)
                    
                    for (page in 1..pagesToProcess) {
                        val pageText = PdfTextExtractor.getTextFromPage(pdfDocument.getPage(page))
                        if (pageText.isNotBlank()) {
                            text.append("## Page $page\n")
                            text.append(pageText.trim())
                            text.append("\n\n")
                        }
                    }
                    
                    pdfDocument.close()
                    
                    val result = text.toString().trim()
                    
                    if (result.isNotBlank()) {
                        // Limit content size to prevent overwhelming the model
                        if (result.length > 15000) {
                            result.take(15000) + "\n\n[Content truncated - showing first 15,000 characters]"
                        } else {
                            result
                        }
                    } else {
                        val fileName = getFileName(context, uri) ?: "document.pdf"
                        "[PDF Document: $fileName - No readable text content found]"
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Error reading PDF file: ${e.message}")
                val fileName = getFileName(context, uri) ?: "document.pdf"
                "[PDF Document: $fileName - Error extracting text: ${e.message}]"
            }
        }
    
    /**
     * Extract text content from Word documents (.docx files are ZIP archives)
     */
    private suspend fun extractWordTextContent(context: Context, uri: Uri): String? = 
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    val bufferedInputStream = inputStream.buffered()
                    
                    // Check if this is a ZIP file (DOCX format)
                    bufferedInputStream.mark(4)
                    val header = ByteArray(4)
                    bufferedInputStream.read(header)
                    bufferedInputStream.reset()
                    
                    if (header.contentEquals(byteArrayOf(0x50, 0x4B, 0x03, 0x04))) {
                        // This is a ZIP file (modern .docx format)
                        extractTextFromDocxZip(bufferedInputStream)
                    } else {
                        // This might be an older .doc file or corrupted
                        val fileName = getFileName(context, uri) ?: "document.doc"
                        "[Legacy Word Document (.doc): $fileName - Please convert to .docx format or export as PDF/TXT for better text extraction]"
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Error reading Word file: ${e.message}")
                val fileName = getFileName(context, uri) ?: "document.docx"
                "[Word Document: $fileName - Error extracting text: ${e.message}]"
            }
        }
    
    /**
     * Extract text content from Excel spreadsheets (.xlsx files are ZIP archives)
     */
    private suspend fun extractExcelTextContent(context: Context, uri: Uri): String? = 
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    val bufferedInputStream = inputStream.buffered()
                    
                    // Check if this is a ZIP file (XLSX format)
                    bufferedInputStream.mark(4)
                    val header = ByteArray(4)
                    bufferedInputStream.read(header)
                    bufferedInputStream.reset()
                    
                    if (header.contentEquals(byteArrayOf(0x50, 0x4B, 0x03, 0x04))) {
                        // This is a ZIP file (modern .xlsx format)
                        extractTextFromExcelZip(bufferedInputStream)
                    } else {
                        // This might be an older .xls file
                        val fileName = getFileName(context, uri) ?: "spreadsheet.xls"
                        "[Legacy Excel File (.xls): $fileName - Please convert to .xlsx format or export as CSV/TXT for better text extraction]"
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Error reading Excel file: ${e.message}")
                val fileName = getFileName(context, uri) ?: "spreadsheet.xlsx"
                "[Excel Spreadsheet: $fileName - Error extracting text: ${e.message}]"
            }
        }
    
    /**
     * Extract text content from PowerPoint presentations (.pptx files are ZIP archives)
     */
    private suspend fun extractPowerPointTextContent(context: Context, uri: Uri): String? = 
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    val bufferedInputStream = inputStream.buffered()
                    
                    // Check if this is a ZIP file (PPTX format)
                    bufferedInputStream.mark(4)
                    val header = ByteArray(4)
                    bufferedInputStream.read(header)
                    bufferedInputStream.reset()
                    
                    if (header.contentEquals(byteArrayOf(0x50, 0x4B, 0x03, 0x04))) {
                        // This is a ZIP file (modern .pptx format)
                        extractTextFromPowerPointZip(bufferedInputStream)
                    } else {
                        // This might be an older .ppt file
                        val fileName = getFileName(context, uri) ?: "presentation.ppt"
                        "[Legacy PowerPoint File (.ppt): $fileName - Please convert to .pptx format or export as PDF/TXT for better text extraction]"
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Error reading PowerPoint file: ${e.message}")
                val fileName = getFileName(context, uri) ?: "presentation.pptx"
                "[PowerPoint Presentation: $fileName - Error extracting text: ${e.message}]"
            }
        }
    
    /**
     * Format file size for display
     */
    fun formatFileSize(bytes: Long): String {
        // Handle edge cases
        if (bytes < 0) {
            android.util.Log.w("FileUtils", "Negative file size: $bytes")
            return "Unknown size"
        }
        if (bytes == 0L) return "0 B"
        if (bytes < 1024) return "$bytes B"
        
        val units = arrayOf("KB", "MB", "GB", "TB")
        var size = bytes.toDouble()
        var unitIndex = 0
        
        while (size >= 1024 && unitIndex < units.size - 1) {
            size /= 1024
            unitIndex++
        }
        
        // Log for debugging large files
        android.util.Log.d("FileUtils", "formatFileSize: $bytes bytes -> %.1f ${units[unitIndex]}".format(size))
        
        return "%.1f %s".format(size, units[unitIndex])
    }
    
    /**
     * Check if file is too large (limit to 10MB for now)
     */
    fun isFileTooLarge(bytes: Long): Boolean {
        return bytes > 10 * 1024 * 1024 // 10MB limit
    }
    
    /**
     * Copy file to internal storage for persistence
     */
    suspend fun copyFileToInternalStorage(context: Context, uri: Uri, filename: String): Uri? = 
        withContext(Dispatchers.IO) {
            try {
                val attachmentsDir = java.io.File(context.filesDir, "attachments")
                if (!attachmentsDir.exists()) {
                    attachmentsDir.mkdirs()
                }
                
                val targetFile = java.io.File(attachmentsDir, "${System.currentTimeMillis()}_$filename")
                
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    targetFile.outputStream().use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
                
                android.util.Log.d("FileUtils", "Copied file to: ${targetFile.absolutePath}")
                Uri.fromFile(targetFile)
            } catch (e: Exception) {
                android.util.Log.e("FileUtils", "Failed to copy file to internal storage", e)
                null
            }
        }
    
    /**
     * Extract text from Word document ZIP archive
     */
    private fun extractTextFromDocxZip(inputStream: java.io.InputStream): String {
        try {
            val zipInputStream = ZipInputStream(inputStream)
            var entry: ZipEntry?
            
            while (zipInputStream.nextEntry.also { entry = it } != null) {
                val entryName = entry?.name ?: continue
                
                if (entryName == "word/document.xml") {
                    val content = zipInputStream.bufferedReader().readText()
                    val extractedText = extractTextFromXML(content)
                    zipInputStream.close()
                    
                    return if (extractedText.isNotBlank()) {
                        // Limit content size to prevent overwhelming the model
                        if (extractedText.length > 15000) {
                            extractedText.take(15000) + "\n\n[Content truncated - showing first 15,000 characters]"
                        } else {
                            extractedText
                        }
                    } else {
                        "[Word Document - No readable text content found]"
                    }
                }
                zipInputStream.closeEntry()
            }
            zipInputStream.close()
            return "[Word Document - No readable text content found]"
        } catch (e: Exception) {
            android.util.Log.e("FileUtils", "Error extracting from Word ZIP: ${e.message}")
            return "[Word Document - Error extracting text from ZIP archive]"
        }
    }
    
    /**
     * Extract text from Excel spreadsheet ZIP archive
     */
    private fun extractTextFromExcelZip(inputStream: java.io.InputStream): String {
        try {
            val zipInputStream = ZipInputStream(inputStream)
            var entry: ZipEntry?
            val worksheetTexts = mutableListOf<String>()
            
            while (zipInputStream.nextEntry.also { entry = it } != null) {
                val entryName = entry?.name ?: continue
                
                // Extract from worksheet files
                if (entryName.startsWith("xl/worksheets/sheet") && entryName.endsWith(".xml")) {
                    val content = zipInputStream.bufferedReader().readText()
                    val extractedText = extractTextFromXML(content)
                    if (extractedText.isNotBlank()) {
                        val sheetNumber = entryName.substringAfter("sheet").substringBefore(".xml")
                        worksheetTexts.add("## Sheet $sheetNumber\n$extractedText")
                    }
                }
                zipInputStream.closeEntry()
            }
            zipInputStream.close()
            
            return if (worksheetTexts.isNotEmpty()) {
                val result = worksheetTexts.joinToString("\n\n")
                // Limit content size to prevent overwhelming the model
                if (result.length > 15000) {
                    result.take(15000) + "\n\n[Content truncated - showing first 15,000 characters]"
                } else {
                    result
                }
            } else {
                "[Excel Spreadsheet - No readable text content found]"
            }
        } catch (e: Exception) {
            android.util.Log.e("FileUtils", "Error extracting from Excel ZIP: ${e.message}")
            return "[Excel Spreadsheet - Error extracting text from ZIP archive]"
        }
    }
    
    /**
     * Extract text from PowerPoint presentation ZIP archive
     */
    private fun extractTextFromPowerPointZip(inputStream: java.io.InputStream): String {
        try {
            val zipInputStream = ZipInputStream(inputStream)
            var entry: ZipEntry?
            val slideTexts = mutableListOf<String>()
            
            while (zipInputStream.nextEntry.also { entry = it } != null) {
                val entryName = entry?.name ?: continue
                
                // Extract from slide files
                if (entryName.startsWith("ppt/slides/slide") && entryName.endsWith(".xml")) {
                    val content = zipInputStream.bufferedReader().readText()
                    val extractedText = extractTextFromXML(content)
                    if (extractedText.isNotBlank()) {
                        val slideNumber = entryName.substringAfter("slide").substringBefore(".xml")
                        slideTexts.add("## Slide $slideNumber\n$extractedText")
                    }
                }
                zipInputStream.closeEntry()
            }
            zipInputStream.close()
            
            return if (slideTexts.isNotEmpty()) {
                val result = slideTexts.joinToString("\n\n")
                // Limit content size to prevent overwhelming the model
                if (result.length > 15000) {
                    result.take(15000) + "\n\n[Content truncated - showing first 15,000 characters]"
                } else {
                    result
                }
            } else {
                "[PowerPoint Presentation - No readable text content found]"
            }
        } catch (e: Exception) {
            android.util.Log.e("FileUtils", "Error extracting from PowerPoint ZIP: ${e.message}")
            return "[PowerPoint Presentation - Error extracting text from ZIP archive]"
        }
    }
    
    /**
     * Simple XML text extraction - removes tags and extracts text content
     */
    private fun extractTextFromXML(xml: String): String {
        return xml
            .replace(Regex("<[^>]*>"), " ") // Remove XML tags
            .replace(Regex("\\s+"), " ") // Normalize whitespace
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .trim()
    }
}
