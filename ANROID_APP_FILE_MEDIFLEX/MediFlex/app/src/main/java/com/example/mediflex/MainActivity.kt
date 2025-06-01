package com.example.mediflex

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import com.example.mediflex.ui.theme.MediFlexTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val dbHelper = DatabaseHelper(this)
        val isUserLoggedIn = dbHelper.isUserLoggedIn()  // Check if user is logged in

        if (isUserLoggedIn) {
            // If logged in, open WelcomeBackActivity
            startActivity(Intent(this, WelcomeBackActivity::class.java))
            finish()  // Prevents going back to Explore screen
        } else {
            // If NOT logged in, show the Welcome Screen
            setContent {
                MediFlexTheme {
                    WelcomeScreen { navigateToAuth() }
                }
            }
        }
    }

    private fun navigateToAuth() {
        startActivity(Intent(this, AuthActivity::class.java))
    }
}

@Composable
fun WelcomeScreen(onExploreClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Welcome to MediFlex",
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onExploreClick,
            modifier = Modifier
                .padding(top = 8.dp)
                .height(50.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF6200EE))
        ) {
            Text(
                text = "Explore",
                fontSize = 18.sp
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun WelcomeScreenPreview() {
    MediFlexTheme {
        WelcomeScreen(onExploreClick = {})
    }
}
