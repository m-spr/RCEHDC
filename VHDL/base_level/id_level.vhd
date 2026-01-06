LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;
    USE std.textio.ALL;

ENTITY idLevel3 IS
    GENERIC (n  : INTEGER := 7;  -- bit-widths input
             c  : INTEGER := 2;  -- coeficient of increasment! c = hv/ 2**n
             r  : INTEGER := 2;  -- remainder from division
             hv : INTEGER := 500 -- hyperdimesional size
            );
    PORT (values   : IN  STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
          idVector : OUT STD_LOGIC_VECTOR(hv - 1 DOWNTO 0));
END ENTITY idLevel3;

ARCHITECTURE Behavioral OF idLevel3 IS

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
    CONSTANT zeros : std_logic_vector(hv - 1 DOWNTO 0) := (OTHERS => '0');
    CONSTANT ones  : std_logic_vector(hv - 1 DOWNTO 0) := (OTHERS => '1');
    SIGNAL idVectorVarS : STD_logic_vector((2 ** n) - 1 DOWNTO 0);
BEGIN

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
        VARIABLE idVectorVar : std_logic_vector((2 ** n) - 1 DOWNTO 0);
    BEGIN
        idVectorVar := (OTHERS => '0');
        IF values = zeros(n - 1 DOWNTO 0) THEN
            idVectorVar := (OTHERS => '0');
        ELSIF values = ones(n - 1 DOWNTO 0) THEN
            idVectorVar := (OTHERS => '1');
        ELSE
            --idVectorVar(hv-1 DOWNTO hv - to_integer(unsigned(values))-1) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
            idVectorVar(to_integer(unsigned(values)) - 1 DOWNTO 0) := ones(to_integer(unsigned(values)) - 1 DOWNTO 0);
        END IF;
        idVectorVarS <= idVectorVar;
    END PROCESS;

    ID_LOOP: FOR I IN C DOWNTO 1 GENERATE
        idVector(((2 ** n) * I) - 1 DOWNTO (2 ** n) * (I - 1)) <= idVectorVarS;
    END GENERATE ID_LOOP;

    remainder_ID: IF r /= 0 GENERATE
        idVector(hv - 1 DOWNTO (2 ** n) * (c)) <= idVectorVarS(r - 1 DOWNTO 0);
    END GENERATE remainder_ID;

END ARCHITECTURE Behavioral;
