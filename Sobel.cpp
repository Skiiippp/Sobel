/**
 * By James Gruber and Daniel Brathwaite, 12/11/2024
 */


#include <cstdio>
#include <string>
#include <cstdint>
#include <chrono>

#include <pthread.h>
#include <semaphore.h>

#include <arm_neon.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#define NUM_THREADS 4

const int x_kern[3][3] = 
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int y_kern[3][3] =
{
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

struct FnameInfo
{
    std::string in_vid_filename;
};

struct ThreadArgument
{
    cv::Mat *input_frame_ptr;
    cv::Mat *grayscale_frame_ptr;
    cv::Mat *sobel_frame_ptr;
    int bottom_row_index;
    size_t rows_to_read;
    bool last;
};

/* Barriers  */
pthread_barrier_t barrier;

/* Threads */
static pthread_t threads[NUM_THREADS];
static ThreadArgument args[NUM_THREADS];

bool sleep_threads = true, kill_threads = false;

void parse_args(int argc, char **argv, struct FnameInfo &fname_info);

void generate_image(cv::Mat &input_frame, cv::Mat &grayscale_frame, cv::Mat &sobel_frame);

void get_grayscale(cv::Mat &input_frame, cv::Mat &grayscale_frame, int lower_row, size_t quantum);

void get_sobel(cv::Mat &grayscale_frame, cv::Mat &sobel_frame, int lower_row, size_t quantum);

uint8_t get_pixel_grayscale(uint8_t red, uint8_t green, uint8_t blue);

/* x and y with respect to sobel, not grayscale */
uint8_t get_pixel_sobel(int x, int y, cv::Mat &grayscale_frame);

void *generate_subset(void *arg);

int main(int argc, char **argv)
{
    cv::Mat input_frame, grayscale_frame, sobel_frame;
    cv::VideoCapture capturer;
    int input_height, input_width;
    cv::Size input_size, output_size;
    bool is_processing_done = false;
    struct FnameInfo fname_info;
    size_t frame_count = 0;
    std::chrono::microseconds avg_frame_durr = std::chrono::microseconds(0), cur_frame_durr;

    parse_args(argc, argv, fname_info);

    if (!capturer.open(fname_info.in_vid_filename))
    {
        printf("Failed to open video capturer\n");
        exit(1);
    }

    input_height = static_cast<int>(capturer.get(cv::CAP_PROP_FRAME_HEIGHT));
    input_width = static_cast<int>(capturer.get(cv::CAP_PROP_FRAME_WIDTH));
    input_size = cv::Size(input_width, input_height);
    output_size = cv::Size(input_width-2, input_height-2);
    
    grayscale_frame = cv::Mat::zeros(input_size, CV_8UC1);
    sobel_frame = cv::Mat::zeros(output_size, CV_8UC1);

    cv::namedWindow("swaos", cv::WINDOW_AUTOSIZE);

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    /* Create threads */
    size_t small_row_quantum = input_frame.rows / NUM_THREADS;
    size_t big_row_quantum = input_frame.rows - (small_row_quantum * (NUM_THREADS - 1));
    struct ThreadArgument *arg;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        arg = &args[i];
        arg->bottom_row_index = i * small_row_quantum;
        arg->rows_to_read = (i == NUM_THREADS-1) ? big_row_quantum : small_row_quantum;
        arg->last = (i == NUM_THREADS-1) ? true : false; 
        arg->input_frame_ptr = &input_frame;
        arg->grayscale_frame_ptr = &grayscale_frame;
        arg->sobel_frame_ptr = &sobel_frame;

        pthread_create(&threads[i], NULL, generate_subset, (void *)arg);
    }

    while (!is_processing_done)
    {
        auto start = std::chrono::steady_clock::now();
        
        frame_count++;
        
        /* Input frames are CV_8UC3*/
        capturer >> input_frame;   

        if (input_frame.empty())
        {
            is_processing_done = true;
            continue;
        }
        
        sleep_threads = false;
        
    
        cv::imshow("swaos", sobel_frame);

        /* Give 1ms to display image */
        if (cv::waitKey(1) >= 0)
        {
            break;
        }
        
        auto end = std::chrono::steady_clock::now();
        
        cur_frame_durr = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg_frame_durr = (((frame_count-1)*avg_frame_durr) + cur_frame_durr)/frame_count;
    }

    capturer.release();
    cv::destroyAllWindows();
    kill_threads = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    pthread_barrier_destroy(&barrier);
    
    auto sec_durr = std::chrono::duration<double>(avg_frame_durr);
    float avg_fps = 1/(sec_durr.count());
    printf("Average FPS: %f\n", avg_fps);

    return 0;
}

void parse_args(int argc, char **argv, struct FnameInfo &fname_info)
{
    if (argc != 2)
    {
        printf("Bad arguments, requires path to input video.\n");
        exit(1);
    }

    fname_info.in_vid_filename = argv[1];
}

void generate_image(cv::Mat &input_frame, cv::Mat &grayscale_frame, cv::Mat &sobel_frame)
{
    size_t small_row_quantum = input_frame.rows / NUM_THREADS;
    size_t big_row_quantum = input_frame.rows - (small_row_quantum * (NUM_THREADS - 1));
    struct ThreadArgument *arg;

    /* Create threads */
    for (int i = 0; i < NUM_THREADS; i++)
    {
        arg = &args[i];
        arg->bottom_row_index = i * small_row_quantum;
        arg->rows_to_read = (i == NUM_THREADS-1) ? big_row_quantum : small_row_quantum;
        arg->last = (i == NUM_THREADS-1) ? true : false; 
        arg->input_frame_ptr = &input_frame;
        arg->grayscale_frame_ptr = &grayscale_frame;
        arg->sobel_frame_ptr = &sobel_frame;

        pthread_create(&threads[i], NULL, generate_subset, (void *)arg);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

void *generate_subset(void *arg)
{
    struct ThreadArgument *thread_arg = (struct ThreadArgument *)arg;
    size_t sobel_quantum = thread_arg->last ? thread_arg->rows_to_read - 2 : thread_arg->rows_to_read;

    while(1)
    {
        while(sleep_threads && !kill_threads);
        if (kill_threads)
        {
            break;
        }
        
        /* Get grayscale */
        get_grayscale(*thread_arg->input_frame_ptr, *thread_arg->grayscale_frame_ptr, thread_arg->bottom_row_index, thread_arg->rows_to_read);

        /* Barrier */
        pthread_barrier_wait(&barrier);

        /* Get sobel */
        get_sobel(*thread_arg->grayscale_frame_ptr, *thread_arg->sobel_frame_ptr, thread_arg->bottom_row_index, sobel_quantum);
        
        pthread_barrier_wait(&barrier);
        sleep_threads = true;
    }

    pthread_exit(NULL);
}

void get_grayscale(cv::Mat &input_frame, cv::Mat &grayscale_frame, int lower_row, size_t quantum)
{
    for (int y = lower_row; y < lower_row + (int)quantum; y++) {
	    for (int x = 0; x < input_frame.cols; x += 4) {
            // Load pixel data
            uint8x8x3_t pixel = vld3_u8(&input_frame.at<cv::Vec3b>(y, x)[0]);

            uint16x8_t red   = vmovl_u8(pixel.val[0]);
            uint16x8_t green = vmovl_u8(pixel.val[1]);
            uint16x8_t blue  = vmovl_u8(pixel.val[2]);

            // Multiply channels by their respective grayscale weights
            uint16x8_t weightedRed   = vmulq_n_u16(red, 54);
            uint16x8_t weightedGreen = vmulq_n_u16(green, 183);
            uint16x8_t weightedBlue  = vmulq_n_u16(blue, 19);

            // Compute normalized sum of weighted channels
            uint16x8_t sum = vaddq_u16(weightedRed, weightedGreen);
            sum = vaddq_u16(sum, weightedBlue);

            uint16x8_t grayscale = vrshrq_n_u16(sum, 8);

            // Convert result back to 8-bit
            uint8x8_t finalGrayscale = vmovn_u16(grayscale);

            vst1_u8(&grayscale_frame.at<uint8_t>(y, x), finalGrayscale);
	    }
    }
}

void get_sobel(cv::Mat &grayscale_frame, cv::Mat &sobel_frame, int lower_row, size_t quantum)
{
    for (int y = lower_row; y < lower_row + (int)quantum; y++)
    {
        for (int x = 1; x < sobel_frame.cols - 1; x += 8)
        {
            // Check that pixel is valid
            if (y - 1 >= 0 && y + 1 < grayscale_frame.rows)
            {
                int16x8_t gx = vdupq_n_s16(0);
                int16x8_t gy = vdupq_n_s16(0);

                // Sobel computation loop
                for (int j = -1; j <= 1; j++) {
                    int8_t weightX = j;
                    int8_t weightY = -j;

                    uint8x8_t row = vld1_u8(&grayscale_frame.at<uint8_t>(y + j, x - 1));
                    
                    // Expand 8bit vals to 16bit to prevent overflow
                    uint16x8_t urow16 = vmovl_u8(row);
                    int16x8_t row16 = vreinterpretq_s16_u16(urow16);

                    // Apply weights
                    gx = vmlaq_n_s16(gx, row16, weightX);
                    gy = vmlaq_n_s16(gy, row16, weightY);
                }

                // Compute magnitude of gradient
                int16x8_t gradient_mag = vaddq_s16(vabsq_s16(gx), vabsq_s16(gy));

                vst1_u8(&sobel_frame.at<uint8_t>(y - 1, x - 1), vqmovun_s16(gradient_mag));
            }
        }
    }
}
