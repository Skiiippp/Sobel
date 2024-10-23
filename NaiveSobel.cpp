/**
 * By James Gruber and Daniel Brathwaite, 10/19/2024
 */


#include <cstdio>
#include <string>
#include <cstdint>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

//const std::string in_vid_filename = "../Megamind.avi";

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

void parse_args(int argc, char **argv, struct FnameInfo &fname_info);

void get_grayscale(cv::Mat &input_frame, cv::Mat &grayscale_frame);

void get_sobel(cv::Mat &grayscale_frame, cv::Mat &sobel_frame);

uint8_t get_pixel_grayscale(uint8_t red, uint8_t green, uint8_t blue);

/* x and y with respect to sobel, not grayscale */
uint8_t get_pixel_sobel(int x, int y, cv::Mat &grayscale_frame);

int main(int argc, char **argv)
{
    cv::Mat input_frame, graysacle_frame, sobel_frame;
    cv::VideoCapture capturer;
    int input_height, input_width;
    cv::Size input_size, output_size;
    bool is_processing_done = false;
    struct FnameInfo fname_info;

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
    
    graysacle_frame = cv::Mat::zeros(input_size, CV_8UC1);
    sobel_frame = cv::Mat::zeros(output_size, CV_8UC1);

    cv::namedWindow("swaos", cv::WINDOW_AUTOSIZE);

    while (!is_processing_done)
    {
        /* Input frames are CV_8UC3*/
        capturer >> input_frame;   

        if (input_frame.empty())
        {
            is_processing_done = true;
            continue;
        }

        get_grayscale(input_frame, graysacle_frame);
        get_sobel(graysacle_frame, sobel_frame);
    
        cv::imshow("swaos", sobel_frame);

        /* Give 1ms to display image */
        if (cv::waitKey(1) >= 0)
        {
            break;
        }
    }

    capturer.release();
    cv::destroyAllWindows();

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

void get_grayscale(cv::Mat &input_frame, cv::Mat &grayscale_frame)
{
    cv::Vec3b input_pixel;

    for (int y = 0; y < input_frame.rows; y++)
    {
        for (int x = 0; x < input_frame.cols; x++)
        {
            input_pixel = input_frame.at<cv::Vec3b>(y,x);
            grayscale_frame.at<uint8_t>(y, x) = get_pixel_grayscale(input_pixel[2], input_pixel[1], input_pixel[0]);
        }
    }
}

void get_sobel(cv::Mat &grayscale_frame, cv::Mat &sobel_frame)
{
    for (int y = 0; y < sobel_frame.rows; y++)
    {
        for (int x = 0; x < sobel_frame.cols; x++)
        {
            sobel_frame.at<uint8_t>(y, x) = get_pixel_sobel(x, y, grayscale_frame);
        }
    }
}

uint8_t get_pixel_grayscale(uint8_t red, uint8_t green, uint8_t blue)
{
    uint16_t pixel_grayscale = 0.2126*red + 0.7152*green + 0.0722*blue;

    if (pixel_grayscale > UINT8_MAX)
    {
        pixel_grayscale = UINT8_MAX;
    }

    return pixel_grayscale;
}

uint8_t get_pixel_sobel(int x, int y, cv::Mat &grayscale_frame)
{
    int grayscale_x = x + 1, grayscale_y = y + 1;
    uint8_t pixel_grayscale;
    int16_t x_grad = 0, y_grad = 0;
    uint16_t x_mag, y_mag;
    uint16_t out_big;

    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            pixel_grayscale = grayscale_frame.at<uint8_t>(grayscale_y + j - 1, grayscale_x + i - 1);
            x_grad += (int16_t)pixel_grayscale * x_kern[i][j];
            y_grad += (int16_t)pixel_grayscale * y_kern[i][j];
        }
    }

    /* Get abs value */
    x_mag = (x_grad < 0) ? x_grad * -1 : x_grad;
    y_mag = (y_grad < 0) ? y_grad * -1 : y_grad;

    out_big = x_mag + y_mag;

    /* Clamp */
    out_big = (out_big > UINT8_MAX) ? UINT8_MAX : out_big;

    return (uint8_t)out_big;
}