#ifndef STATISTICS_ML__CONVOLUTION_H_
#define STATISTICS_ML__CONVOLUTION_H_

#include <armadillo>
#include <ros/ros.h>

namespace stats{

template <typename T>
inline void conv3b(arma::colvec3& boarders,
                   const double delta_vol,
                   const arma::Cube<T>& P,
                   const arma::Col<T>& kernel_x,
                   const arma::Col<T>& kernel_y,
                   const arma::Col<T>& kernel_z){

    ROS_INFO_STREAM_THROTTLE(1.0,"convolution check");

    int M       = kernel_x.n_elem;
    int k       = (M-1)/2; // middle element of the kernel
    int stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    boarders.zeros();

    ROS_INFO_STREAM_THROTTLE(1.0,"k: " << k << " M: " << M );

    // X
    std::size_t r     = 0;
    std::size_t r_end = P.n_rows-1;
    std::size_t d = 0;
    std::size_t c = 0;

    double tmp1,tmp2;

    for(d=0; d < P.n_slices;d++){
       for(c=0; c < P.n_cols;c++){
           tmp1 = 0;
           tmp2 = 0;
            for(int m = 0; m < stop; m++){
                if(r+m >= 0 && r+m <= r_end && r_end - m >= 0){
                    //std::cout<< "P("<<r + m<<","<<c<<","<<d<<") :" << P(r+m,c,d) * delta_vol << std::endl;
                   //  std::cout<< "P("<<r + m<<","<<c<<","<<d<<")  x K(" << m + k + 1 << ") "<<std::endl;
                    tmp1 = tmp1 + P(r + m,c,d)     * delta_vol  * kernel_x(k + m + 1);
                    tmp2 = tmp2 + P(r_end - m,c,d) * delta_vol  * kernel_x(k - m - 1);
                }
           }
           boarders(0) = boarders(0) + tmp1/(P(r,c,d) * delta_vol) + tmp2/(P(r_end,c,d) * delta_vol);
       }
    }


    M       = kernel_y.n_elem;
    k       = (M-1)/2;
    stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    ROS_INFO_STREAM("d conv y");
    std::cout<< "k: " << k << std::endl;

    // Y
    c=0;
    std::size_t c_end = P.n_cols-1;
    for(d=0; d < P.n_slices;d++){
        for(r=0; r < P.n_rows;r++){
            tmp1 = 0;
            tmp2 = 0;
            for(int m = 0; m < stop; m++){
                if(c+m >= 0 && c+m <= c_end && c_end - m >= 0){
                //    std::cout<< "P("<<r<<","<<c+m<<","<<d<<") x K(" << m + k + 1 << ") +" << " P(" <<r<< "," << c_end - m <<","<<d<<") x K(" << k - 1 - m << ") "<< std::endl;

                    tmp1 = tmp1 + P(r,c+m,d)       * delta_vol  * kernel_y(k + m + 1);
                    tmp2 = tmp2 + P(r,c_end - m,d) * delta_vol  * kernel_y(k - m - 1);
                }
            }
            boarders(1) = boarders(1) + tmp1/(P(r,c,d) * delta_vol) + tmp2/(P(r,c_end,d) * delta_vol);
        }
    }


    M       = kernel_z.n_elem;
    k       = (M-1)/2;
    stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    ROS_INFO_STREAM("d conv z");
    // Z
    d     = 0;
    std::size_t d_end = P.n_slices-1;
    for(c=0; c < P.n_cols;c++){
        for(r=0; r < P.n_rows;r++){
            tmp1 = 0;
            tmp2 = 0;
            for(int m = 0; m < stop; m++){
                if(d+m >= 0 && d+m <= d_end && d_end - m >= 0){
                    tmp1 = tmp1 + P(r,c,d+m)       * delta_vol   * kernel_z(k + m + 1);
                    tmp2 = tmp2 + P(r,c,d_end - m) * delta_vol   * kernel_z(k - m - 1);
                }
            }
            boarders(2) = boarders(2) + tmp1/(P(r,c,d) * delta_vol) + tmp2/(P(r,c,d_end) * delta_vol);

        }
    }


}


template <typename T>
inline void conv3(arma::Cube<T>& Pn ,
                  const arma::Cube<T>& P,
                  const arma::Col<T>& kernel_x,
                  const arma::Col<T>& kernel_y,
                  const arma::Col<T>& kernel_z){

    ROS_INFO_STREAM_THROTTLE(1.0,"convolution");

    int M       = kernel_x.n_elem;
    int k       = (M-1)/2;
    int stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    ROS_INFO_STREAM_THROTTLE(1.0,"k: " << k << " M: " << M );

    // X
    for(std::size_t d=0; d < P.n_slices;d++){
        for(std::size_t c=0; c < P.n_cols;c++){

            for(std::size_t r=0; r < P.n_rows;r++){
                for(int m = -k; m <= stop; m++){
                    if(r+m >= 0 && r+m < P.n_rows){
                        Pn(r,c,d) = Pn(r,c,d) + P(r + m,c,d) * kernel_x(m + k);
                    }

                }
            }

        }
    }


    M       = kernel_y.n_elem;
    k       = (M-1)/2;
    stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    ROS_INFO_STREAM("conv y");

    // Y
    for(std::size_t d=0; d < P.n_slices;d++){
        for(std::size_t r=0; r < P.n_rows;r++){
            for(std::size_t c=0; c < P.n_cols;c++){

                for(int m = -k; m <= stop; m++){
                    if(c+m >= 0 && c+m < P.n_cols){
                        Pn(r,c,d) = Pn(r,c,d) + P(r,c+m,d) * kernel_y(m + k);
                    }
                }


            }
        }
    }

    M       = kernel_z.n_elem;
    k       = (M-1)/2;
    stop    = k;
    if(M == 2)
    {
        k    = 0;
        stop = 1;
    }

    ROS_INFO_STREAM("conv z");
    // Z
    for(std::size_t c=0; c < P.n_cols;c++){
        for(std::size_t r=0; r < P.n_rows;r++){
            for(std::size_t d=0; d < P.n_slices;d++){


                for(int m = -k; m <= stop; m++){
                    if(d+m >= 0 && d+m < P.n_slices){
                        Pn(r,c,d) = Pn(r,c,d) + P(r,c,d+m) * kernel_z(m + k);
                    }
                }
            }
        }
    }


}


}


#endif
