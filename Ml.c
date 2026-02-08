#define STB_IMAGE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include "stb_image.h"
#include <time.h>
#include <sys/stat.h>


#define ITERATIONS 1000
#define WEIGHT_DECAY 0.0001
#define NUM_TRIALS 5
#define batch_size 4
#define BETA1 0.9
#define BETA2 0.999 
#define EPSILON 1e-4
#define TRAIN_RATIO 0.8
#define N 20
#define V ((N * N) + 1)
#define IMAGE_SIZE 400
#define DATASET_SIZE 400
#define EPOCHS 100
#define LEARNING_RATE 0.01
#define TRAIN_SIZE (int)(DATASET_SIZE * TRAIN_RATIO) // Eðitim seti boyutu 
#define TEST_SIZE (DATASET_SIZE - TRAIN_SIZE) // Test seti boyutu





typedef struct {//piksel ve etikete ulaþmak için yapý
    float pixels[IMAGE_SIZE + 1];
    int label;
} Image;

void read_pgm(const char *filename, float *buffer) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Dosya açýlamadý: %s\n", filename);
        exit(1);
    }

    char magic_number[3];
    int width, height, max_val;
    fscanf(file, "%s\n%d %d\n%d\n", magic_number, &width, &height, &max_val);
unsigned char *temp = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    fread(temp, sizeof(unsigned char), width * height, file);
    fclose(file);
	int i;
    for ( i = 0; i < width * height; i++) {
        buffer[i] = temp[i] / 255.0f;
    }
    free(temp);
}
void shuffle_images(Image *images, int num_images) {
	int i,j;
    for ( i = 0; i < num_images; i++) {
        j = rand() % num_images;
        Image temp = images[i];
        images[i] = images[j];
        images[j] = temp;
    }
}

void create_datasets_from_directory(const char *directory, int label, Image **images, int *num_images, int *capacity) {
    struct dirent *entry;
    DIR *dp = opendir(directory);
    if (dp == NULL) {
        perror("opendir");
        exit(1);
    }

    while ((entry = readdir(dp))) {
        if (entry->d_name[0] != '.') { 
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);
            
            struct stat statbuf;
            if (stat(filepath, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) { 
                if (*num_images >= *capacity) {
                    *capacity *= 2;
                    *images = (Image*)realloc(*images, *capacity * sizeof(Image));
                }

                read_pgm(filepath, (*images)[*num_images].pixels);
                (*images)[*num_images].label = label; 

                (*num_images)++;
            }
        }
    }

    closedir(dp);
}





float tanh_activation(float x) {//tanh ý hesaplar
 return tanh(x); 
 }

 float tanh_derivative(float x) { //tanh ýn türevini hesaplar
 	float tanh_x = tanh(x);
 	 return 1 - tanh_x * tanh_x; 
 }
void initialize_weights(float *w, int size,double value) {//aðýrlýk deðerlerini girer
	int i;
	//float j=((float)rand() / (float)(RAND_MAX)) * 2.0 - 1.0;
 for (i = 0; i < size; ++i) { 
 w[i] = value; // w[i] =j;
}
}
void save_weights(const char *filename, float *w, int size,int epoch) { 
	FILE *file = fopen(filename, "a");
	int i; 
	if (file == NULL) { 
	printf("Dosya açýlamadý: %s\n", filename); 
	return; 
	} 
	fprintf(file, "Epoch %d:\n",epoch); 
	for (i = 0; i < size; ++i) { 
	fprintf(file, "%f\n", w[i]); 
	} 
	fprintf(file, "\n"); // Her epoch sonrasý boþluk býrak fclose(file);
	fclose(file);
	}
void gradient_descent(Image *train_set, Image *test_set, int train_size, int test_size, float *w, float *train_losses, float *test_losses, float *train_accuracies, float *test_accuracies,float *times) {
    int epoch, i, j;
	float learning_rate = 0.1;
	clock_t start,end; 	  //zamaný ölçmek için tanýmlamalar
	const char *filename = "gdweights1.txt";
    for (epoch = 0; epoch < EPOCHS; ++epoch) { //100 iterasyona sahip EPOCHS'u yukarýda tanýmladýk
        float total_loss = 0.0; 
        float gradients[V] = {0.0}; //gradientlarý her iterasyonda sýfýrlýyoruz
        start = clock();// her iterasyonun süresini hesaplýyoruz

        for (i = 0; i < train_size; ++i) {
            Image img = train_set[i];//eðitim kümesindeki her görseli kullanýyoruz
            float y_true = img.label;//img  yapýsýndan görselin sýnýfýný alýyoruz
            float y_pred = 0.0;

            
            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];//aðýrlýk deðerleri ve piksellerin çarpýmý ile ypred deðiþkenini elde ediyoruz
            }
            y_pred = tanh_activation(y_pred);//tanh modelinde y_pred deðerini kullanarak label'ý tahmin ediyoruz

            
            float error = y_true - y_pred;//gerçek deðer ile tahmini çýkartarak hatamýzý ölçüyoruz
            total_loss += error * error; 

            
            for (j = 0; j < V; ++j) {
                gradients[j] += -2.0 * error * tanh_derivative(y_pred) * img.pixels[j];//her piksel deðeri için gradientlar hesaplanýr
            }
        }

      
        for (j = 0; j < V; ++j) {
            w[j] -= learning_rate * gradients[j] / train_size;//hesapladýðýmýz gradientlarý kullanarak aðýrlýk deðerlerini güncelleriz
        }

        train_losses[epoch] = total_loss / train_size;//iterasyondaki loss deðerlerini kaydederiz
		
		//eðitim kümesinin doðruluðunu hesaplama
        int correct_predictions = 0;
        for (i = 0; i < train_size; ++i) {
            float y_pred = 0.0;
            for (j = 0; j < V; ++j) {
                y_pred += w[j] * train_set[i].pixels[j];
            }
            y_pred = tanh_activation(y_pred);
            if ((y_pred >= 0 && train_set[i].label == 1) || (y_pred < 0 && train_set[i].label == -1)) {
                correct_predictions++;
            }
        }
        train_accuracies[epoch] = (float)correct_predictions / train_size;
		
        float total_test_loss = 0.0;
        float acc = 0.0;
        for (i = 0; i < test_size; ++i) {//yukarýda eðitim kümesi için yaptýklarýmýz test içinde yaparýz
            Image img = test_set[i];
            float y_true = img.label;
            float y_pred = 0.0;

            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];
            }
            y_pred = tanh_activation(y_pred);

            float error = y_true - y_pred;
            total_test_loss += error * error;
            int predicted_label = (y_pred > 0) ? 1 : -1;
            if (predicted_label == y_true) {
                acc++;
            }
        }

        test_accuracies[epoch] = acc / test_size;
        test_losses[epoch] = total_test_loss / test_size;
   		end = clock(); // zamaný durdurur ve kaydederiz
		times[epoch] = ((float)(end - start)) / CLOCKS_PER_SEC;
		//save_weights(filename, w,V,epoch);
   }	
   
}
/*void results (const char *method,float *train_losses, float *test_losses, float *train_accuracies, float *test_accuracies){
	int epoch;
	
	for (epoch=0;epoch<EPOCHS;epoch++){
	    printf("Epoch %d   Train Loss %f   Test Loss %f   Eðitim Doðruluðu %f   Test Doðruluðu %f\n",
        epoch, train_losses[epoch], test_losses[epoch], train_accuracies[epoch], test_accuracies[epoch]);
		
	}
}*/
void stochastic_gradient_descent(Image *train_set, Image *test_set, int train_size, int test_size, float *w, float *train_losses, float *test_losses, float *train_accuracies, float *test_accuracies,float *times) {
    int epoch, i, j, randIndex;
    float learning_rate = 0.1;
    clock_t start,end;
	const char *filename = "sgdweights1.txt";
    for (epoch = 0; epoch < EPOCHS; ++epoch) {
        start = clock();
		float total_loss = 0.0;
		randIndex = rand() % (int)(train_size);//veriyi  rastgele seçeriz
        Image img = train_set[randIndex];
        for (i = 0; i < train_size;i+=batch_size) {//verinin belli bir miktar böleriz batch size ile
            float y_true = img.label;
            float y_pred = 0.0;
			
          
            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];
            }
            y_pred = tanh_activation(y_pred);

           
            float error = y_true - y_pred;
            total_loss += error * error;

            for (j = 0; j < V; ++j) {
                float gradient = -2.0 * error * tanh_derivative(y_pred) * img.pixels[j];//gradient descent ile farký bir diziye kayýt etmediðimizden daha hýzlý iþliyor
                w[j] -= learning_rate * gradient;
            }
        }

        train_losses[epoch] = total_loss / train_size;

        int correct_predictions = 0;
        for (i = 0; i < train_size; ++i) {
            float y_pred = 0.0;
            for (j = 0; j < V; ++j) {
                y_pred += w[j] * train_set[i].pixels[j];
            }
            y_pred = tanh_activation(y_pred);
            if ((y_pred >= 0 && train_set[i].label == 1) || (y_pred < 0 && train_set[i].label == -1)) {
                correct_predictions++;
            }
        }
        train_accuracies[epoch] = (float)correct_predictions / train_size;
		train_losses[epoch] = total_loss / train_size;   		
		
        float total_test_loss = 0.0;
        float acc = 0.0;
        for (i = 0; i < test_size; ++i) {
            Image img = test_set[i];
            float y_true = img.label;
            float y_pred = 0.0;

            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];
            }
            y_pred = tanh_activation(y_pred);

            float error = y_true - y_pred;
            total_test_loss += error * error;
            int predicted_label = (y_pred > 0) ? 1 : -1;
            if (predicted_label == y_true) {
                acc++;
            }
        }
		
        test_accuracies[epoch] = acc / test_size;
        test_losses[epoch] = total_test_loss / test_size;
		end = clock(); 
		times[epoch] = ((float)(end - start)) / CLOCKS_PER_SEC;
		
		//save_weights(filename, w,V,epoch);
    }
}

void adam_optimizer(Image *train_set, Image *test_set, int train_size, int test_size, float *w, float *train_losses, float *test_losses, float *train_accuracies, float *test_accuracies,float *times) {
    int epoch, i, j, t, randIndex;
    float mt[V] = {0.0};
    float vt[V] = {0.0};
    float beta1 = BETA1;
    float beta2 = BETA2;
    float epsilon = EPSILON;
    float learning_rate = LEARNING_RATE;
	clock_t start,end;
	const char *filename = "adamweights1.txt";
    for (epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0;
        start = clock();

        for (i = 0; i < train_size; i+=2*batch_size) {
            randIndex = rand() % (int)(train_size);
            Image img = train_set[randIndex];
            float y_true = img.label;
            float y_pred = 0.0;

            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];
            }
            y_pred = tanh_activation(y_pred);

            float error = y_true - y_pred;
            total_loss += error * error;

            for (j = 0; j < V; ++j) {
                float gradient = -2.0 * error * tanh_derivative(y_pred) * img.pixels[j];
                mt[j] = beta1 * mt[j] + (1 - beta1) * gradient;//momentum bir önceki veriyi kullanarak daha dengeli bir güncelleme saðlar
                vt[j] = beta2 * vt[j] + (1 - beta2) * gradient * gradient;//momentumdan farký gradientýn karesini alýr ve bu sayede büyük sapmalarý azaltýr

                float mt_hat = mt[j] / (1 - pow(beta1, epoch + 1));//baþlangýçta düþük deðerler olacaðýndan bunu engellemek için kullanýlýr
                float vt_hat = vt[j] / (1 - pow(beta2, epoch + 1));

                w[j] -= learning_rate * mt_hat / (sqrt(vt_hat) + epsilon);
            }
        }

        train_losses[epoch] = total_loss / train_size;

        int correct_predictions = 0;
        for (i = 0; i < train_size; ++i) {
            float y_pred = 0.0;
            for (j = 0; j < V; ++j) {
                y_pred += w[j] * train_set[i].pixels[j];
            }
            y_pred = tanh_activation(y_pred);
            if ((y_pred >= 0 && train_set[i].label == 1) || (y_pred < 0 && train_set[i].label == -1)) {
                correct_predictions++;
            }
        }
        train_accuracies[epoch] = (float)correct_predictions / train_size;
        train_losses[epoch] = total_loss / train_size;

        float total_test_loss = 0.0;
        float acc = 0.0;
        for (i = 0; i < test_size; ++i) {
            Image img = test_set[i];
            float y_true = img.label;
            float y_pred = 0.0;

            for (j = 0; j < V; ++j) {
                y_pred += w[j] * img.pixels[j];
            }
            y_pred = tanh_activation(y_pred);

            float error = y_true - y_pred;
            total_test_loss += error * error;
            int predicted_label = (y_pred > 0) ? 1 : -1;
            if (predicted_label == y_true) {
                acc++;
            }
        }

        test_accuracies[epoch] = acc / test_size;
        test_losses[epoch] = total_test_loss / test_size;
		end = clock(); 
		times[epoch] = ((float)(end - start)) / CLOCKS_PER_SEC;
		//save_weights(filename, w,V,epoch);
    }
}



float evaluate_accuracy(float *weights, Image *dataset, int dataset_size) {
    int correct_predictions = 0;
    int i, j;

    for (i = 0; i < dataset_size; i++) {
        float dot_product = 0.0;

        for (j = 0; j < 401; j++) {
            dot_product += weights[j] * dataset[i].pixels[j];
        }

        float prediction = tanh_activation(dot_product);

        int predicted_label = (prediction > 0) ? 1 : -1;

        if (predicted_label == dataset[i].label) {
            correct_predictions++;
        }
    }

    return (float)correct_predictions / dataset_size;
}



   
void results(const char *method, float *train_losses, float *test_losses, float *train_accuracies, float *test_accuracies,float value,float *times,float *weights) {
    int epoch;
    char filename[50];
    sprintf(filename, "%f_%s.txt",value, method);//lost, accuracy, time verilerini tutmak için dosya
    FILE *file = fopen(filename, "w");
    //sprintf(filename, "agirlik%f_%s.txt",value, method);//aðýrlýk deðerlerini tutmak için dosya 
    //FILE *file1 = fopen(filename, "w");
    if(file != NULL) {
        for (epoch = 0; epoch < EPOCHS; epoch++) {
            printf("Epoch %d   Train Loss %f   Test Loss %f   Eðitim Doðruluðu %f   Test Doðruluðu %f zaman %f\n",
                   epoch, train_losses[epoch], test_losses[epoch], train_accuracies[epoch], test_accuracies[epoch],times[epoch]);

			fprintf(file, "Epoch %d\n", epoch + 1);
            fprintf(file, "Train Loss: %f\n", train_losses[epoch]);
            fprintf(file, "Test Loss: %f\n", test_losses[epoch]);
            fprintf(file, "Train Accuracy: %f\n", train_accuracies[epoch]);
            fprintf(file, "Test Accuracy: %f\n", test_accuracies[epoch]);
            fprintf(file, "Süre: %f\n", times[epoch]);
        }
       /* for(epoch =0 ; epoch<V-1;epoch++){
        	fprintf(file1, "%f ",weights[epoch]);
			if(epoch%4==3){
			fprintf(file1, "\n");
		}
		
		}
		fclose(file1);*/
		
        fclose(file);
        
    } 
	else {
        printf("Dosya oluþturulamadý: %s\n", filename);
    }
}



void evaluate_model(const char *method, Image *train_set, int train_size, Image *test_set, int test_size, void (*optimizer)(Image *,Image *, int,int, float *, float *,float *,float *,float *,float *), double value) {
    float w[V];
    float train_losses[EPOCHS];
    float test_losses[EPOCHS];
    float train_accuracies[EPOCHS];
    float test_accuracies[EPOCHS];
	float times[EPOCHS];
    
    int i, j, epoch;
    int length = EPOCHS;

    initialize_weights(w, V, value);// baþlangýç aðýrlýk deðerlerini yazar 
	
    optimizer(train_set, test_set, train_size, test_size, w, train_losses, test_losses, train_accuracies, test_accuracies,times);//algoritmayý çaðýrýr
   
  results(method,train_losses, test_losses, train_accuracies, test_accuracies,value,times,w);// dosya iþlemlerini yapmak için gereken fonksiyonu çaðýrýr

   
	//çok bir iþe yaramýyorlar ama býraktým son doðruluk deðerlerini yazmak için fonksiyon çaðýrýr
    float test_accuracy = evaluate_accuracy(w, test_set, test_size);
    float train_accuracy = evaluate_accuracy(w, train_set, train_size);
    printf("%s Eðitim Doðruluðu: %.2f, Test Doðruluðu: %.2f\n", method, 100*train_accuracy, 100*test_accuracy);
}












int main() {
    const char *directory1 = "C:\\Users\\faruk\\Desktop\\arþift\\diff part1\\tro";
    const char *directory2 = "C:\\Users\\faruk\\Desktop\\arþift\\diff part1\\shr"; 
	int i,trial;
    int capacity = 10;
    Image *images = (Image*)malloc(capacity * sizeof(Image));
    int num_images = 0;
	float weights[V];
	
	
    create_datasets_from_directory(directory1, 1, &images, &num_images, &capacity);  
    create_datasets_from_directory(directory2, -1, &images, &num_images, &capacity); 
	shuffle_images(images, num_images);
    int train_size = num_images * 0.8;
    int test_size = num_images - train_size;
	int negatif=0,pozitif=0;
	

    Image *train_set = images;
    Image *test_set = images + train_size;
	printf("Training Set Labels:\n");
	 for ( i = 0; i < train_size; i++) {//data setin daðýlýmýný kontrol eder
	 	 printf("Image %d: Label %d\n", i, train_set[i].label);
	  	 } 
	   	 printf("Test Set Labels:\n");
	  	 for ( i = 0; i < test_size; i++) {
		 printf("Image %d: Label %d\n", i, test_set[i].label);
		 if(test_set[i].label==1){
		 	pozitif+=1;
		 }
		 else{
		 	negatif+=1;
		 }
		 
	}
	printf("test elmalar %d\ntest kediler %d\n",pozitif,negatif);
    printf("Total images: %d\n", num_images);
    printf("Training set size: %d\n", train_size);
    printf("Test set size: %d\n", test_size);
	double value=10.0;
 for ( trial = 0; trial < NUM_TRIALS; ++trial) {// 5 tane aðýrlýk deðerinde algoritmalarý dener
        printf("\nDeneme %d:\n", trial + 1);
		value/=10;//((float)rand() / (float)(RAND_MAX)) * 0.2 - 0.1;
        evaluate_model("Gradient Descent", train_set, train_size, test_set, test_size, gradient_descent,value);
        
        
        evaluate_model("Stochastic Gradient Descent", train_set, train_size, test_set, test_size, stochastic_gradient_descent,value);

      evaluate_model("Adam Optimizer", train_set, train_size, test_set, test_size, adam_optimizer,value);
      if(value == 0.1){
      	
	  }
    }

   
    free(images);

    return 0;
}


