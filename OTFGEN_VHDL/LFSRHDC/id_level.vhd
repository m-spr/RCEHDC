LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use std.textio.all; 

entity idLevel3 is
    GENERIC (n : INTEGER := 7;		 		 	-- bit-widths input
			 c 	: INTEGER := 2; 				-- coeficient of increasment! c = hv/ 2**n
			 r  : INTEGER := 2;                  -- remainder from division
			 hv : INTEGER := 500		 	 	-- hyperdimesional size
			 );
	Port ( values : in STD_LOGIC_VECTOR (n-1 DOWNTO 0);
           idVector : out STD_LOGIC_VECTOR (hv-1 DOWNTO 0));
end idLevel3;

architecture Behavioral of idLevel3 is

--CONSTANT zeros : std_logic_vector (hv-1 DOWNTO 0) := (others => '0');
--CONSTANT ones : std_logic_vector (hv-1 DOWNTO 0) := (others => '1');
--begin

--PROCESS (values)
--VARIABLE idVectorVar : std_logic_vector (hv-1 DOWNTO 0);
--begin
--	idVectorVar := (others => '0');
--	if (values = ones(n-1 DOWNTO 0)) then
--		idVectorVar := ones;
--	elsif (values = zeros(n-1 DOWNTO 0)) then
--		idVectorVar := zeros;
--	else
--		if c = 1 then
--		l_parity0 : for k in 1 TO hv-1 loop
--			if (k < to_integer(unsigned(values)+1))then
--				idVectorVar(hv-k) := ones(0);
--			else
--				exit;
--			end if;
--			end loop;
--		else
--		l_parity : for k in 1 TO hv-1 loop
--			if (k < to_integer(unsigned(values)))then
--				idVectorVar((hv-((k-1)*c))-1 downto hv-((k)*c)) := ones(c-1 DOWNTO 0);
--			else
--				exit;
--			end if;
--			end loop;
--		end if;
--	end if;

--	idVector <= idVectorVar;
--end PROCESS;
CONSTANT zeros : std_logic_vector (hv-1 DOWNTO 0) := (others => '0');
CONSTANT ones : std_logic_vector (hv-1 DOWNTO 0) := (others => '1');
SIGNAL idVectorVarS : STD_logic_vector ((2**n)-1 DOWNTO 0);
begin



--PROCESS (values)
--VARIABLE idVectorVar : std_logic_vector (hv-1 DOWNTO 0);
--begin
--	idVectorVar := (others => '0');
--	if values = zeros(n-1 DOWNTO 0) then
--		idVectorVar := (others => '0');
--	elsif to_integer(unsigned(values)) = hv then
--		idVectorVar := (others => '1');
--	else
--		--idVectorVar(hv-1 DOWNTO hv - to_integer(unsigned(values))-1) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
--		idVectorVar(to_integer(unsigned(values))-1 DOWNTO 0) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
--	end if;

PROCESS (values)
VARIABLE idVectorVar : std_logic_vector ((2**n)-1 DOWNTO 0);
begin
	   idVectorVar := (others => '0');
	if values = zeros(n-1 DOWNTO 0) then
		idVectorVar := (others => '0');
	elsif values = ones(n-1 DOWNTO 0) then
		idVectorVar := (others => '1');
	else
		--idVectorVar(hv-1 DOWNTO hv - to_integer(unsigned(values))-1) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
		idVectorVar(to_integer(unsigned(values))-1 DOWNTO 0) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
	end if;
idVectorVarS <= idVectorVar;	
end PROCESS;

 ID_LOOP: FOR I IN C DOWNTO 1 GENERATE
		idVector(((2**n)*I)-1 Downto (2**n)*(I-1)) <= idVectorVarS;
	END GENERATE ID_LOOP;
	
remainder_ID : if r /= 0 generate
    idVector(hv-1 Downto (2**n)*(c)) <= idVectorVarS( r-1  Downto  0);
  end generate remainder_ID;


end Behavioral;
