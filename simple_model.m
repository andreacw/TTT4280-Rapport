load('muab.mat');


red_wavelength = 600;
green_wavelength = 515;
blue_wavelength = 460;

wavelengths = [red_wavelength, green_wavelength, blue_wavelength];


mua_blood_oxy = @(x) interp1(muabo(:,1), muabo(:,2), x);
mua_blood_deoxy = @(x) interp1(muabd(:,1), muabd(:,2), x);


bvf = 0.01; %blood volume fraction, average amount of blood in the tissue
oxy = 0.8; %oxygenation of the blood

%absorption coefficient (mu_a in lab text)
%units: m^(-1)
mua_other = 25; %background absorption due to collagen etc
mua_blood = mua_blood_oxy(wavelengths)*oxy + mua_blood_deoxy(wavelengths)*(1-oxy); %absorption due to pure blood
mua = mua_blood*bvf + mua_other;

%reduced scattering coefficient (mu_s' in lab text)
%the magic numbers are from N. Bashkatov, E. A. Genina, V. V. Tuchin.
%Optical properties of skin, subcutaneous and muscle tissues: A review. J.
%Inoov.  Opt. Health Sci., 4(1):9-38, 2011
%units: m^(-1)
musr = (17.6*(wavelengths/500.0).^(-4) + 18.78*(wavelengths/500).^(-0.22))*100;

%mua and musr are now available as three-valued arrays, each index corresponding to: red, green and blue channels.


%%INSERT CODE FOR CALCULATING PENETRATION DEPTH DEL
pddr = sqrt(1/(3*(musr(1) + mua(1))*mua(1)));       %Penetration depth del (red)
pddg = sqrt(1/(3*(musr(2) + mua(2))*mua(2)));       %Penetration depth del (green)
pddb = sqrt(1/(3*(musr(3) + mua(3))*mua(3)));       %Penetration depth del (blue)

ft = 17/1000; %[m], finger thickness

red = pddr*ft;
green = pddg*ft;
blue = pddb*ft;



