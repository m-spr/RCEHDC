-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE STD.TEXTIO.ALL; 

ENTITY idLevel3 IS
    GENERIC (
        n  : INTEGER := 7;  -- Bit-width of input
        c  : INTEGER := 2;  -- Coefficient of increment (hv / 2^n)
        r  : INTEGER := 2;  -- Remainder from division
        hv : INTEGER := 500 -- Hyperdimensional size
    );
    PORT (
        values   : IN  STD_LOGIC_VECTOR(n-1 DOWNTO 0);
        idVector : OUT STD_LOGIC_VECTOR(hv-1 DOWNTO 0)
    );
END ENTITY idLevel3;

ARCHITECTURE Behavioral OF idLevel3 IS

    -- Constants for Zero and One Values
    CONSTANT zeros : STD_LOGIC_VECTOR(hv-1 DOWNTO 0) := (OTHERS => '0');
    CONSTANT ones  : STD_LOGIC_VECTOR(hv-1 DOWNTO 0) := (OTHERS => '1');

    -- Signal to Hold Intermediate ID Vector
    SIGNAL idVectorVarS : STD_LOGIC_VECTOR((2**n)-1 DOWNTO 0);

BEGIN

    -- Process to Compute `idVectorVarS`
    PROCESS (values)
        VARIABLE idVectorVar : STD_LOGIC_VECTOR((2**n)-1 DOWNTO 0);
    BEGIN
        -- Default Initialization
        idVectorVar := (OTHERS => '0');

        -- Assign based on input values
        IF values = zeros(n-1 DOWNTO 0) THEN
            idVectorVar := (OTHERS => '0');
        ELSIF values = ones(n-1 DOWNTO 0) THEN
            idVectorVar := (OTHERS => '1');
        ELSE
            -- Assign a range of ones based on `values`
            idVectorVar(to_integer(unsigned(values))-1 DOWNTO 0) := 
                ones(to_integer(unsigned(values))-1 DOWNTO 0);
        END IF;

        -- Assign to the signal
        idVectorVarS <= idVectorVar;
    END PROCESS;

    -- Generate Full Copies of `idVectorVarS`
    ID_LOOP: FOR I IN C DOWNTO 1 GENERATE
        idVector(((2**n)*I)-1 DOWNTO (2**n)*(I-1)) <= idVectorVarS;
    END GENERATE ID_LOOP;

    -- Assign Remaining Bits for `r`
    remainder_ID: IF r /= 0 GENERATE
        idVector(hv-1 DOWNTO (2**n)*c) <= idVectorVarS(r-1 DOWNTO 0);
    END GENERATE remainder_ID;

END ARCHITECTURE Behavioral;
