package com.mzm.sample.digit_recognizer

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * This image classifier classifies each drawing as one of the 10 digits
 */
class Classifier @Throws(IOException::class)
constructor(private val context: Context) {

    private val interpreter: Interpreter

    init {
        val assetManager = context.assets
        val model = loadTFModelFile(assetManager)

        val options = Interpreter.Options()
        //options.setNumThreads(3)
        options.setUseNNAPI(true)

        interpreter = Interpreter(model, options)
    }

    @Throws(IOException::class)
    private fun loadTFModelFile(assetManager: AssetManager): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(TF_MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * To classify an image, follow these steps:
     * 1. pre-process the input image
     * 2. run inference with the model
     * 3. post-process the output result for displaying in UI
     *
     */
    fun classify(bitmap: Bitmap): String {
        val inputByteBuffer = preprocess(bitmap)
        val outputArray = Array(BATCH_SIZE) { FloatArray(DIGITS) }
        interpreter.run(inputByteBuffer, outputArray)
        return postprocess(outputArray)
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, false)
        return convertBitmapToByteBuffer(scaledBitmap)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4
                * BATCH_SIZE    // 1
                * INPUT_WIDTH   // 28
                * INPUT_HEIGHT  // 28
                * PIXEL_SIZE)   // 1
        byteBuffer.order(ByteOrder.nativeOrder())

        val imagePixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        bitmap.getPixels(imagePixels, 0, bitmap.width, 0, 0,
                bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until INPUT_WIDTH) {
            for (j in 0 until INPUT_HEIGHT) {
                val `val` = imagePixels[pixel++]
                byteBuffer.putFloat(convertToGreyScale(`val`))
            }
        }
        return byteBuffer
    }

    private fun convertToGreyScale(color: Int): Float {
        val r = (color shr 16 and 0xFF).toFloat()
        val g = (color shr 8 and 0xFF).toFloat()
        val b = (color and 0xFF).toFloat()

        val grayscaleValue = (0.299f * r + 0.587f * g + 0.114f * b).toInt()
        return grayscaleValue / 255.0f
    }

    private fun postprocess(outputArray: Array<FloatArray>): String {
        // Index with highest probability
        var maxIndex = -1
        var maxProb = 0.0f
        for (i in 0 until outputArray[0].size) {
            if (outputArray[0][i] > maxProb) {
                maxProb = outputArray[0][i]
                maxIndex = i
            }
        }
        val resultString =
            "Prediction Number : %d\nConfidence: %2f"
                .format(maxIndex, outputArray[0][maxIndex])

        return resultString
    }

    companion object {
        private val TF_MODEL_PATH = "mnist.tflite"

        // Input size
        private val BATCH_SIZE = 1    // batch size
        private val INPUT_WIDTH = 28  // input image width
        private val INPUT_HEIGHT = 28 // input image height
        private val PIXEL_SIZE = 1    // 1 for gray scale & 3 for color images

        // Output size is 10 (number of digits)
        private val DIGITS = 10
    }

}
