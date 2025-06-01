package com.example.mediflex

import android.content.ContentValues
import android.content.Context
import android.content.SharedPreferences
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class DatabaseHelper(context: Context) : SQLiteOpenHelper(context, "mediflex.db", null, 1) {
    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences("MediflexPrefs", Context.MODE_PRIVATE)

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE users (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "name TEXT, " +
                    "email TEXT UNIQUE, " +
                    "password TEXT)"
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS users")
        onCreate(db)
    }

    fun registerUser(name: String, email: String, password: String): Boolean {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("name", name)
            put("email", email)
            put("password", password)
        }

        val result = db.insert("users", null, values)
        return result != -1L
    }

    fun loginUser(email: String, password: String): String? {
        val db = readableDatabase
        val query = "SELECT name FROM users WHERE email=? AND password=?"
        val cursor = db.rawQuery(query, arrayOf(email, password))

        var userName: String? = null
        if (cursor.moveToFirst()) {
            userName = cursor.getString(0)  // ✅ Get the username from DB
        }
        cursor.close()

        return userName // ✅ Return username if found, null otherwise
    }

    fun setUserLoggedIn(email: String, name: String) {
        sharedPreferences.edit().apply {
            putBoolean("isLoggedIn", true)
            putString("userEmail", email)
            putString("userName", name)  // ✅ Store username for quick access
            apply()
        }
    }

    fun isUserLoggedIn(): Boolean {
        return sharedPreferences.getBoolean("isLoggedIn", false)
    }

    fun getLoggedInUser(): String? {
        return sharedPreferences.getString("userEmail", null)
    }

    fun getLoggedInUserName(): String? {
        return sharedPreferences.getString("userName", null)  // ✅ Faster lookup
    }

    fun logout() {
        sharedPreferences.edit().clear().apply()
    }
}
