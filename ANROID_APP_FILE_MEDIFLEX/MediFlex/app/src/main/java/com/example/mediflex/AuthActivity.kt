package com.example.mediflex

import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class AuthActivity : AppCompatActivity() {
    private lateinit var dbHelper: DatabaseHelper
    private var isLoginMode = true  // Tracks login/register mode

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login_register)

        dbHelper = DatabaseHelper(this)

        val nameInput = findViewById<EditText>(R.id.name_input)
        val emailInput = findViewById<EditText>(R.id.email_input)
        val passwordInput = findViewById<EditText>(R.id.password_input)
        val passwordToggle = findViewById<ImageView>(R.id.password_toggle)
        val loginButton = findViewById<Button>(R.id.login_button)
        val registerButton = findViewById<Button>(R.id.register_button)
        val switchText = findViewById<TextView>(R.id.switch_text)

        // Initially, show login mode
        nameInput.visibility = View.GONE
        registerButton.visibility = View.GONE

        // Switch between login and register mode
        switchText.setOnClickListener {
            isLoginMode = !isLoginMode
            if (isLoginMode) {
                nameInput.visibility = View.GONE
                registerButton.visibility = View.GONE
                loginButton.visibility = View.VISIBLE
                switchText.text = "Don't have an account? Register"
            } else {
                nameInput.visibility = View.VISIBLE
                registerButton.visibility = View.VISIBLE
                loginButton.visibility = View.GONE
                switchText.text = "Already have an account? Login"
            }
        }

        // ✅ Password Visibility Toggle (Fix)
        var isPasswordVisible = false
        passwordToggle.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            if (isPasswordVisible) {
                passwordInput.inputType = InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
                passwordToggle.setImageResource(R.drawable.ic_visibility)  // Show open eye icon
            } else {
                passwordInput.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
                passwordToggle.setImageResource(R.drawable.ic_visibility_off)  // Show closed eye icon
            }
            passwordInput.setSelection(passwordInput.text.length)
        }

        // ✅ Handle Login
        loginButton.setOnClickListener {
            val email = emailInput.text.toString().trim()
            val password = passwordInput.text.toString().trim()

            if (email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "Please enter email and password", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val userName = dbHelper.loginUser(email, password)
            if (userName != null) {
                dbHelper.setUserLoggedIn(email, userName)  // ✅ Save both email & username

                Toast.makeText(this, "Login Successful!", Toast.LENGTH_SHORT).show()
                startActivity(Intent(this, WelcomeBackActivity::class.java))
                finish()
            } else {
                Toast.makeText(this, "Invalid email or password!", Toast.LENGTH_SHORT).show()
            }
        }

        // ✅ Handle Registration
        registerButton.setOnClickListener {
            val name = nameInput.text.toString().trim()
            val email = emailInput.text.toString().trim()
            val password = passwordInput.text.toString().trim()

            if (name.isEmpty() || email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "Please fill all fields!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val success = dbHelper.registerUser(name, email, password)
            if (success) {
                dbHelper.setUserLoggedIn(email, name)  // ✅ Save login state
                Toast.makeText(this, "Registration Successful!", Toast.LENGTH_SHORT).show()
                startActivity(Intent(this, WelcomeBackActivity::class.java))
                finish()
            } else {
                Toast.makeText(this, "Email already registered!", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
