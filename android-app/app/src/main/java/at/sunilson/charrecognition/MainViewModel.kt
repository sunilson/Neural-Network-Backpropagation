package at.sunilson.charrecognition

import android.app.Application
import android.app.ProgressDialog
import android.arch.lifecycle.AndroidViewModel
import android.arch.lifecycle.LiveData
import android.arch.lifecycle.ViewModel
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.Environment
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import io.reactivex.Observable
import io.reactivex.ObservableEmitter
import io.reactivex.ObservableOnSubscribe
import io.reactivex.Scheduler
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.schedulers.Schedulers
import okhttp3.MediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Retrofit
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Callable

/**
 * @author Linus Weiss
 */



class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val app : Application = application
    val gson = GsonBuilder().setLenient().create()
    val retrofit = Retrofit.Builder().baseUrl("http://10.0.2.2:5000").addCallAdapterFactory(RxJava2CallAdapterFactory.create()).build().create(RetrofitService::class.java)

    override fun onCleared() {
        super.onCleared()
    }

    fun recognizeChar(bitmap: Bitmap): Observable<Char> {

        return Observable.fromCallable{createFileFromBitmap(bitmap)}.subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).flatMap {
            val reqFile: RequestBody = RequestBody.create(MediaType.parse("image/*"), it)
            val body: MultipartBody.Part = MultipartBody.Part.createFormData("image", it.name, reqFile)
            val requestBody: RequestBody = RequestBody.create(MultipartBody.FORM, "upload_test")
            retrofit.charRecognition(body, requestBody).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).map {
                it.string().single()
            }
        }
    }

    fun createFileFromBitmap(bitmap: Bitmap) : File {
        val f = File(Environment.getExternalStorageDirectory().absolutePath,"image.png" )
        val fos = FileOutputStream(f)
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
        fos.flush()
        fos.close()
        return f
    }
}