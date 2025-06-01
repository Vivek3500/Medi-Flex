package com.example.mediflex

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class WelcomeBackActivity : AppCompatActivity() {
    private lateinit var dbHelper: DatabaseHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome_back)

        // Enable Options Menu
        supportActionBar?.setDisplayHomeAsUpEnabled(false)

        // ✅ Initialize DatabaseHelper
        dbHelper = DatabaseHelper(this)

        // Fetch username and display it safely
        val usernameText = findViewById<TextView>(R.id.welcome_username_text)
        val loggedInUserName = dbHelper.getLoggedInUserName()?.takeIf { it.isNotEmpty() } ?: "User"

        usernameText.text = "Welcome Back, $loggedInUserName"

        // Find Medicines Button Click Handling
        findViewById<Button>(R.id.search_button).setOnClickListener {
            startActivity(Intent(this, MedicineRecommendationActivity::class.java))
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.profile_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.profile_info -> {
                startActivity(Intent(this, ProfileActivity::class.java))
                return true
            }
            R.id.logout -> {
                dbHelper.logout()  // ✅ Clear User Login Data
                Toast.makeText(this, "Logged out successfully!", Toast.LENGTH_SHORT).show()

                Intent(this, AuthActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                    startActivity(this)
                }
                finish()
                return true
            }
        }
        return super.onOptionsItemSelected(item)
    }
}
