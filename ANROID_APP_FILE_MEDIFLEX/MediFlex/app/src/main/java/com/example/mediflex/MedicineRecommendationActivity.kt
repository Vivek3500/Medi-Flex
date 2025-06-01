package com.example.mediflex

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody

class MedicineRecommendationActivity : AppCompatActivity() {
    private val client = OkHttpClient()

    private lateinit var symptomsInput: EditText
    private lateinit var searchButton: Button
    private lateinit var resultsTextView: TextView
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_medicine_recommendation)

        symptomsInput = findViewById(R.id.symptoms_input)
        searchButton = findViewById(R.id.search_button)
        resultsTextView = findViewById(R.id.results_text)
        progressBar = findViewById(R.id.progress_bar)

        searchButton.setOnClickListener {
            val symptoms = symptomsInput.text.toString().trim()
            if (symptoms.isNotEmpty()) {
                getMedicineRecommendations(symptoms)
            } else {
                Toast.makeText(this, "Please enter symptoms!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // This helps in connecting with the backend (server)
    private fun getMedicineRecommendations(symptoms: String) {
        val url = "http://10.0.2.2:8080/predict/" // port used for android ip
        val json = JSONObject().apply {
            put("symptoms", symptoms)  // Send JSON format
        }

        val body = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    resultsTextView.text = "Error: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseData = response.body?.string()
                if (response.isSuccessful && responseData != null) {
                    runOnUiThread {
                        displayMedicines(responseData)
                    }
                } else {
                    runOnUiThread {
                        resultsTextView.text = "No recommendations found!"
                    }
                }
            }
        })
    }


    private fun displayMedicines(responseData: String) {
        try {
            val jsonResponse = JSONObject(responseData)
            val medicinesArray: JSONArray = jsonResponse.optJSONArray("recommended_medicines") ?: JSONArray()

            if (medicinesArray.length() == 0) {
                resultsTextView.text = "No medicines found for the given symptoms."
                return
            }

            val resultString = StringBuilder("Recommended Medicines:\n")
            for (i in 0 until medicinesArray.length()) {
                val medicine = medicinesArray.optJSONObject(i)
                val name = medicine?.optString("name", "Unknown Medicine") ?: "Unknown Medicine"
                val confidence = (medicine?.optDouble("confidence", 0.0) ?: 0.0) * 100 // Convert to percentage

                resultString.append("$name - Confidence: ${"%.2f".format(confidence)}%\n")
            }

            resultsTextView.text = resultString.toString()
        } catch (e: Exception) {
            resultsTextView.text = "Error processing response."
        }
    }
}
