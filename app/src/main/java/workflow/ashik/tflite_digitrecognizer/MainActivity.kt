package workflow.ashik.tflite_digitrecognizer

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.mzm.sample.digit_recognizer.Classifier
import java.io.IOException
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            classifier = Classifier(this)
        } catch (e: IOException){
            e.printStackTrace()
        }

        classify_btn.setOnClickListener {
            val bitmap = drawnumber?.getBitmap()
            if(bitmap!=null)
            {
                val digit = classifier.classify(bitmap)
                prediction_txt!!.text = digit
            }
        }

        reset_btn.setOnClickListener {
            drawnumber?.reset()
            prediction_txt!!.text = ""
        }
    }
}
