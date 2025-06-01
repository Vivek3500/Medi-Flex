package com.example.mediflex

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class ProfileActivity : AppCompatActivity() {
    private lateinit var dbHelper: DatabaseHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_profile)

        dbHelper = DatabaseHelper(this)

        val nameText = findViewById<TextView>(R.id.name_text)
        val emailText = findViewById<TextView>(R.id.email_text)

        // ✅ Fetch user details
        val userEmail = dbHelper.getLoggedInUser()
        val userName = dbHelper.getLoggedInUserName()

        nameText.text = "Name: $userName"
        emailText.text = "Email: $userEmail"
    }
}
