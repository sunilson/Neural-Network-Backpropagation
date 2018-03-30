package at.sunilson.charrecognition

import android.app.ProgressDialog
import android.arch.lifecycle.ViewModelProviders
import android.content.pm.PackageManager
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.observers.DisposableObserver
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*


class MainActivity : AppCompatActivity() {

    val composite: CompositeDisposable = CompositeDisposable()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)

        val viewModel = ViewModelProviders.of(this).get(MainViewModel::class.java)

        fab.setOnClickListener { view ->

            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
            } else {
                val progressDialog = ProgressDialog(this, R.style.AppCompatAlertDialogStyle)
                progressDialog.setCancelable(false)
                progressDialog.setTitle("Getting prediction")
                progressDialog.setMessage("Loading...")
                progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER)
                progressDialog.isIndeterminate = true
                progressDialog.show()
                composite.add(viewModel.recognizeChar(ink_view.bitmap).subscribeWith(object : DisposableObserver<Char>() {
                    override fun onComplete() {
                    }

                    override fun onNext(t: Char) {
                        Toast.makeText(this@MainActivity, "Guessed the digit " + t, Toast.LENGTH_LONG).show()
                        progressDialog.hide()
                    }

                    override fun onError(e: Throwable) {
                        Toast.makeText(this@MainActivity, e.message, Toast.LENGTH_LONG).show()
                        progressDialog.hide()
                    }

                }))
            }
        }

        ink_view.setColor(resources.getColor(android.R.color.black))
        ink_view.setMinStrokeWidth(16f)
        ink_view.setMaxStrokeWidth(20f)
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.menu_erase -> {
                ink_view.clear()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        composite.dispose()
    }
}