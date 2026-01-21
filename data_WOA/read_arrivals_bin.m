function [ Arr, Pos ] = read_arrivals_bin( ARRFile )
% Read the BINARY format arrivals data file written by BELLHOP
% (Standard implementation needed to parse the .arr file)

marker_len = 1; % For gfortran usually
fid = fopen( ARRFile, 'r' );

if ( fid == -1 )
   error( 'Arrivals file cannot be opened' );
end

fseek( fid, 4 * marker_len, 0 );
flag = fread( fid, [ 1, 4 ], 'uint8=>char' );

if ~strcmp( flag, '''2D''' ) && ~strcmp( flag, '''3D''' )
   % Sometimes there is no quote
   if strcmp(char(flag), '2D  ') || strcmp(char(flag), '3D  ')
       % handle slightly different format
   else
       fclose(fid);
       error( 'Not a BINARY format Arrivals file?' );
   end
end

if strcmp( flag, '''2D''' ) || strcmp(char(flag), '2D  ')
   fseek( fid, 8 * marker_len, 0 );
   Pos.freq = fread( fid, 1,   'float32' );
   
   fseek( fid, 8 * marker_len, 0 );
   Nsz      = fread( fid, 1,   'int32'   );
   Pos.s.z  = fread( fid, Nsz, 'float32' );
   
   fseek( fid, 8 * marker_len, 0 );
   Nrz      = fread( fid, 1,   'int32'   );
   Pos.r.z  = fread( fid, Nrz, 'float32' );
   
   fseek( fid, 8 * marker_len, 0 );
   Nrr      = fread( fid, 1,   'int32'   );
   Pos.r.r  = fread( fid, Nrr, 'float32' );
   
   Arr = repmat( struct( 'Narr', { int16(0) }, 'A', { single(1.0i) }, 'delay', { single(1.0i) } ), Nrr, Nrz, Nsz );
   
   for isd = 1 : Nsz
      fseek( fid, 8 * marker_len, 0 );
      Narrmx2 = fread( fid, 1, 'int32' );
      for irz = 1 : Nrz
         for irr = 1 : Nrr
            fseek( fid, 8 * marker_len, 0 );
            Narr = fread( fid, 1, 'int32' );
            Arr( irr, irz, isd ).Narr = int16( Narr );
            if Narr > 0
               da = fread( fid, [ 8 + 2 * marker_len, Narr ], '*single' );
               da = da( 2 * marker_len + 1 : end, 1 : Narr );
               Arr( irr, irz, isd ).A = single( da( 1, : ) .* exp( 1.0i * da( 2, : ) * pi/180.0 ) );
               Arr( irr, irz, isd ).delay = single( da( 3, : ) + 1.0i * da( 4, : ) );
            end
         end
      end
   end
end
fclose( fid );
end