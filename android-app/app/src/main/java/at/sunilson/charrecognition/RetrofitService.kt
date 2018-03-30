package at.sunilson.charrecognition

import io.reactivex.Observable
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import java.util.*


interface RetrofitService {
    @Multipart
    @POST("/")
    fun charRecognition(@Part image: MultipartBody.Part, @Part("description") description: RequestBody): Observable<ResponseBody>
}