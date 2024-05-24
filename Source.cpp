#include <iostream>
#include <opencv2/opencv.hpp>
#include <aes.h>
#include <filters.h>
#include <modes.h>
#include <osrng.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include "cryptlib.h"
#include "rijndael.h"
#include "modes.h"
#include "files.h"
#include "osrng.h"
#include "hex.h"

// Baylasan Al-Badawi 211169
// Renal Salah 227556

using namespace std;
using namespace cv;
using namespace CryptoPP;
using namespace std::chrono;

void encryptImage(const string& inputImagePath, const string& outputImagePath, const SecByteBlock& key, const byte* iv) {
    Mat image = imread(inputImagePath, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return;
    }

    vector<byte> plaintext(image.datastart, image.dataend);
    vector<byte> ciphertext(plaintext.size());

    CBC_Mode<AES>::Encryption encryption;
    encryption.SetKeyWithIV(key, key.size(), iv);

    ArraySource(plaintext.data(), plaintext.size(), true,
        new StreamTransformationFilter(encryption, new ArraySink(ciphertext.data(), ciphertext.size())));

    Mat encryptedImage(image.size(), image.type(), ciphertext.data());
    imwrite(outputImagePath, encryptedImage);

    imshow("Encrypted Image", encryptedImage);
    waitKey(0);
}
void decryptImage(const string& inputImagePath, const string& outputImagePath, const SecByteBlock& key, const byte* iv) {
    Mat encryptedImage = imread(inputImagePath, IMREAD_GRAYSCALE);

    if (encryptedImage.empty()) {
        cerr << "Error: Could not open or find the encrypted image." << endl;
        return;
    }

    vector<byte> ciphertext(encryptedImage.datastart, encryptedImage.dataend);
    vector<byte> decryptedtext(ciphertext.size());

    CBC_Mode<AES>::Decryption decryption;
    decryption.SetKeyWithIV(key, key.size(), iv);

    ArraySource(ciphertext.data(), ciphertext.size(), true,
        new StreamTransformationFilter(decryption, new ArraySink(decryptedtext.data(), decryptedtext.size())));

    Mat decryptedImage(encryptedImage.size(), encryptedImage.type(), decryptedtext.data());
    imwrite(outputImagePath, decryptedImage);

    imshow("Decrypted Image", decryptedImage);
    waitKey(0);
}

void hideInformation(Mat& image, const string& info) {
    if (image.empty()) {
        cerr << "Error: Image is empty." << endl;
        return;
    }

    int infoIndex = 0;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (infoIndex >= info.size() * 8) {
                return;
            }

            uchar pixelValue = image.at<uchar>(i, j);

            uchar bit = (info[infoIndex / 8] >> (infoIndex % 8)) & 1;

            pixelValue = (pixelValue & 0xFE) | bit;

            image.at<uchar>(i, j) = pixelValue;

            ++infoIndex;
        }
    }
}

string retrieveInformation(const Mat& image, int length) {
    if (image.empty()) {
        cerr << "Error: Image is empty." << endl;
        return "";
    }

    string info(length, '\0');
    int infoIndex = 0;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (infoIndex >= length * 8) {
                break;
            }

            uchar pixelValue = image.at<uchar>(i, j);

            uchar bit = pixelValue & 1;

            info[infoIndex / 8] |= (bit << (infoIndex % 8));

            ++infoIndex;
        }
    }
    return info;
}

double calculateNPCR(const Mat& original, const Mat& encrypted) {
    int diffCount = 0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            if (original.at<uchar>(i, j) != encrypted.at<uchar>(i, j)) {
                ++diffCount;
            }
        }
    }
    return (diffCount * 100.0) / (original.rows * original.cols);
}

double calculateUACI(const Mat& original, const Mat& encrypted) {
    double totalDiff = 0.0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            totalDiff += abs(original.at<uchar>(i, j) - encrypted.at<uchar>(i, j));
        }
    }
    return (totalDiff / (original.rows * original.cols * 255.0)) * 100;
}

double calculateHammingDistance(const Mat& original, const Mat& encrypted) {
    int totalBits = original.rows * original.cols * 8;
    int differingBits = 0;

    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            uchar orig = original.at<uchar>(i, j);
            uchar enc = encrypted.at<uchar>(i, j);
            for (int k = 0; k < 8; ++k) {
                if (((orig >> k) & 1) != ((enc >> k) & 1)) {
                    ++differingBits;
                }
            }
        }
    }
    return (differingBits * 100.0) / totalBits;
}

double chiSquareTest(const Mat& image) {
    vector<int> hist(256, 0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            hist[image.at<uchar>(i, j)]++;
        }
    }

    double chiSquare = 0.0;
    double expected = image.rows * image.cols / 256.0;
    for (int i = 0; i < 256; ++i) {
        chiSquare += pow(hist[i] - expected, 2) / expected;
    }
    return chiSquare;
}

Mat drawHistogram(const Mat& image) {
    vector<int> hist(256, 0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            hist[image.at<uchar>(i, j)]++;
        }
    }

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255));

    int max_value = *max_element(hist.begin(), hist.end());
    for (int i = 0; i < 256; ++i) {
        hist[i] = ((double)hist[i] / max_value) * histImage.rows;
    }

    for (int i = 1; i < 256; ++i) {
        line(histImage, Point(bin_w * (i - 1), hist_h - hist[i - 1]),
            Point(bin_w * i, hist_h - hist[i]), Scalar(0), 2, 8, 0);
    }

    return histImage;
}

double correlationAnalysis(const Mat& image, bool plot = false) {
    double sum = 0.0, sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
    int n = image.rows * (image.cols - 1);

    vector<Point> points;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols - 1; ++j) {
            int x = image.at<uchar>(i, j);
            int y = image.at<uchar>(i, j + 1);

            points.push_back(Point(x, y));

            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;
        }
    }

    if (plot) {
        Mat plotImage = Mat::zeros(300, 300, CV_8UC3);
        for (const auto& point : points) {
            plotImage.at<Vec3b>(point.y, point.x) = Vec3b(255, 255, 255);
        }
        imshow("Correlation Plot", plotImage);
        waitKey(0);
    }

    double corr = (n * sumXY - sumX * sumY) / sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return corr;
}

double informationEntropy(const Mat& image) {
    vector<int> hist(256, 0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            hist[image.at<uchar>(i, j)]++;
        }
    }

    double entropy = 0.0;
    int totalPixels = image.rows * image.cols;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / totalPixels;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

void measureEncryptionTime(const string& inputImagePath, const string& outputImagePath, const SecByteBlock& key, const byte* iv) {
    auto start = high_resolution_clock::now();

    encryptImage(inputImagePath, outputImagePath, key, iv);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    cout << "Encryption Time (ET): " << duration << " ms" << endl;

    Mat image = imread(inputImagePath, IMREAD_GRAYSCALE);
    int totalPixels = image.rows * image.cols;
    cout << "Encryption Speed (NCPB): " << (double)duration / totalPixels << " ms/pixel" << endl;
}

void displayMenu() {
    cout << "1. Encrypt Image" << endl;
    cout << "2. Hide Information in Image" << endl;
    cout << "3. Retrieve Information from Image" << endl;
    cout << "4. Calculate NPCR" << endl;
    cout << "5. Calculate UACI" << endl;
    cout << "6. Calculate Hamming Distance" << endl;
    cout << "7. Perform Chi-Square Test" << endl;
    cout << "8. Perform Correlation Analysis" << endl;
    cout << "9. Calculate Information Entropy" << endl;
    cout << "10. Measure Encryption Time" << endl;
    cout << "11. Decrypt Image" << endl;
    cout << "12. Exit" << endl;
    cout << "Enter your choice: ";
}

int main() {
    string inputImagePath = "C:/Users/MSI/Downloads/LenaRGB.bmp";
    string encryptedImagePath = "C:Users/MSI/source/repos/AESProjectBaylasanRenal/Encrypted_image.jpg";
    string decryptedImagePath = "Decrypted_image.jpg";

    AutoSeededRandomPool prng;
    SecByteBlock key(AES::DEFAULT_KEYLENGTH);
    byte iv[AES::BLOCKSIZE];

    prng.GenerateBlock(key, key.size());
    prng.GenerateBlock(iv, sizeof(iv));

    Mat originalImage = imread(inputImagePath, IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return -1;
    }

    while (true) {
        displayMenu();
        int choice;
        cin >> choice;

        switch (choice) {
        case 1:
            encryptImage(inputImagePath, encryptedImagePath, key, iv);
            break;
        case 2: {
            string confidentialInfo;
            cout << "Enter information to hide: ";
            cin >> confidentialInfo;
            hideInformation(originalImage, confidentialInfo);
            string hiddenInfoImagePath = "Hidden_info_image.jpg";
            imwrite(hiddenInfoImagePath, originalImage);
            cout << "Information hidden in the image and saved to " << hiddenInfoImagePath << endl;
            break;
        }
        case 3: {
            int length;
            cout << "Enter the length of information to retrieve: ";
            cin >> length;
            string retrievedInfo = retrieveInformation(originalImage, length);
            cout << "Retrieved Information: " << retrievedInfo << endl;
            break;
        }
        case 4: {
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "NPCR: " << calculateNPCR(originalImage, encryptedImage) << "%" << endl;
            break;
        }
        case 5: {
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "UACI: " << calculateUACI(originalImage, encryptedImage) << "%" << endl;
            break;
        }
        case 6: {
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "Hamming Distance: " << calculateHammingDistance(originalImage, encryptedImage) << "%" << endl;
            break;
        }
        case 7: {
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "Chi-square Test: " << chiSquareTest(encryptedImage) << endl;
            break;
        }
        case 8: {
            cout << "Performing Correlation Analysis..." << endl;
            cout << "Original Image Correlation: " << correlationAnalysis(originalImage, true) << endl;
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "Encrypted Image Correlation: " << correlationAnalysis(encryptedImage, true) << endl;
            break;
        }
        case 9: {
            Mat encryptedImage = imread(encryptedImagePath, IMREAD_GRAYSCALE);
            cout << "Information Entropy: " << informationEntropy(encryptedImage) << endl;
            break;
        }
        case 10:
            measureEncryptionTime(inputImagePath, encryptedImagePath, key, iv);
            break;
        case 11:
            decryptImage(encryptedImagePath, decryptedImagePath, key, iv);
            break;
        case 12:
            cout << "Exiting..." << endl;
            return 0;
        default:
            cout << "Invalid choice. Please try again." << endl;
        }
    }

    return 0;
}
