function [C, WVN, x, y, z]=load_hack
filename='/bph/puredata1/bioinfdata/user/arnrau/Daten/NN_FTIR_Data/140710_e38735_Arne/140710_e38735.pure'
 fid=fopen(filename);
 fseek(fid,0, 'bof');
 ver=fread(fid,[1 21],'*char');
 if strcmp(ver,'Pure file version 0.1')
    pointer(1)=fread(fid,1,'double');
    pointer(2)=fread(fid,1,'double');
    pointer(3)=fread(fid,1,'double');
    fseek(fid,9*8+pointer(1), 'bof');
    x=fread(fid,1,'double');
    y=fread(fid,1,'double');
    z=fread(fid,1,'double');
    fseek(fid,pointer(2), 'bof');
    WVN=fread(fid,z,'double');
    if isnan(WVN(1))
        WVN=[];
    end
    fseek(fid,pointer(3), 'bof');
    C=fread(fid,x*y*z,'double');
    if y>1 && x>1
       C=reshape(C,x,y,z);
    else
        C=reshape(C,x*y,z);
    end
    fclose(fid);
 else
    disp('File not compatible with this importer version.'); 
    fclose(fid);
 end
end


