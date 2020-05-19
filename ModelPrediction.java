import org.datavec.api.split.FileSplit;
import org.datavec.image.data.Image;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

public class ModelPrediction {
    public static void main(String[] args) throws IOException {
        long height=28;
        long width=28;
        long deepth=1;
        long minBatch=1;
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("digitRecognition.zip"));
        File predict=new File("C:\\Users\\Lenovo\\Desktop\\BDCC-S4\\SMA\\CNN-digit\\src\\main\\resources\\mnist_png\\mnist_png\\Prediction\\30.png");
        ImageLoader imageLoader=new ImageLoader();
        Image img=imageLoader.asImageMatrix(predict);
        INDArray imgData=img.getImage().reshape(minBatch,deepth,height,width);
        INDArray prediction=model.output(imgData);
        String[] labels={"0","1","2","3","4","5","6","7","8","9"};
        int[] data=prediction.argMax(1).toIntVector();
        System.out.println("The number in the picture is : "+labels[data[0]]);


    }
}
