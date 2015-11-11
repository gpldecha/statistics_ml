/*
 * distributions.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: guillaume
 */

#include <statistics/distributions/distributions.h>
#include <boost/lexical_cast.hpp>







///////////////////////////
//    	Epanechnikov     //
///////////////////////////

Epanechnikov::Epanechnikov(int nx,double cx,int Ns):gen(boost::variate_generator<ENG,DIST>(ENG(),DIST(-1,1))){
	this->nx = nx;
	this->cx = cx;
	this->Ns = Ns;

	A = std::pow(8*(1/cx)*(nx + 4)*std::pow((2*std::sqrt(M_PI)),nx),1/(nx + 4));
	h = A* std::pow((double)Ns,1/(nx + 4));

}

void Epanechnikov::SetMean(const arma::vec& x_mean){
	this->x_mean = x_mean;
    this->x_mean.resize(x_mean.n_elem);
    x.resize(x_mean.n_elem);
}

void Epanechnikov::SetCov(const arma::mat& D){
	this->D = D;
}

inline double Epanechnikov::K(double u){
	if(fabs(u) < 1){
		return 0.75*( 1 - u*u);
	}else{
		return 0;
	}
}

arma::vec Epanechnikov::sample(){
	for(int i = 0; i < 3; i++){
		e[i] = K(gen());
	}

	h = 1;
	x = D*e*h + x_mean ;
    return x;
}


///////////////////////////
//    	Uniform			 //
///////////////////////////


Uniform::Uniform():generator(static_cast<unsigned int>(std::time(0))){
    origin.zeros();
    orient.resize(3,3);
    orient.eye();
};

Uniform::Uniform(const arma::vec3& origin,const arma::mat& orient,double length, double width, double height):
	origin(origin),
	orient(orient),
	length(length),
	width(width),
	height(height),
	generator(static_cast<unsigned int>(std::time(0)))
{

	init();

}

void Uniform::init(){
	double lower,upper;
    upper =  length/2.0;
    lower =  -length/2.0;
	uniform_x =   boost::uniform_real<double>(lower,upper);


    upper =  width/2;
    lower =  -width/2;

	uniform_y = boost::uniform_real<double>(lower,upper);

    upper =  height/2;
    lower =  -height/2;

	uniform_z = boost::uniform_real<double>(lower,upper);
}

void Uniform::SetOrigin(const arma::vec& origin){
	this->origin = origin;
	init();
}

void Uniform::Set(double length,double width,double height){
	this->length=length;
	this->width=width;
	this->height=height;
	init();
}

arma::vec Uniform::sample() {
    arma::vec tmp(3);
    tmp(0) = uniform_x(generator);
    tmp(1) = uniform_y(generator);
    tmp(2) = uniform_z(generator);
    tmp = orient * tmp + origin;
    return tmp;
}

arma::vec Uniform::GetOrigin(){
	return origin;
}

double Uniform::getLength(){
	return length;
}

double Uniform::getWidth(){
	return width;
}

double Uniform::getHeight(){
	return height;
}

void Uniform::draw(){
 //TODO: draw the probability distributions
}

void Uniform::translate(const arma::vec& T){
	origin = origin + T;
	init();
}



///////////////////////////
//  Mixture Of Uniform	 //
///////////////////////////

MixUniform::MixUniform(){}

MixUniform::~MixUniform(){
}

MixUniform::MixUniform(const std::vector<Uniform>& uniformList,const std::vector<double>& P):
uniformList(uniformList),
dist(P)
{
    dist = boost::random::discrete_distribution<>(P);
}

arma::vec MixUniform::sample(){
    unsigned int index = dist(generator);
    //std::cout<< "index: " << index << std::endl;
    //std::cout<< "uniformList.size(): " << uniformList.size() << std::endl;
    return uniformList[index].sample();
}

void MixUniform::translate(const arma::vec& T){
	for(unsigned int i = 0; i < uniformList.size(); i++){
		uniformList[i].translate(T);
	}

}
