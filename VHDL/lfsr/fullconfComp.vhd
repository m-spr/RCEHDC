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

ENTITY fullconfComp IS
    GENERIC (
        n            : INTEGER := 10;   -- Bit-width for memory pointer, counter, etc.
        classNumber  : INTEGER := 10;   -- Total number of classes
        classPortion : INTEGER := 10    -- Portion size per class
    );
    PORT (
        clk          : IN  STD_LOGIC;                          -- Clock
        rst          : IN  STD_LOGIC;                          -- Reset
        run          : IN  STD_LOGIC;                          -- Run control (should remain '1' during computation)
        done         : IN  STD_LOGIC;                          -- Done signal from control
        hv           : IN  STD_LOGIC;                          -- Hypervector input
        Chv_input    : IN  STD_LOGIC;                          -- Class hypervector input (bitwise)
        pointer      : IN  STD_LOGIC_VECTOR(n - 1 DOWNTO 0);   -- Pointer/index input
        sim          : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0)    -- Similarity output
    );
END ENTITY fullconfComp;

ARCHITECTURE behavioral OF fullconfComp IS

    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Output width of popCounter
        );
        PORT (
            clk   : IN STD_LOGIC;
            rst   : IN STD_LOGIC;
            en    : IN STD_LOGIC;
            dout  : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT reg IS
        GENERIC (
            lenPop : INTEGER := 8  -- Width of register
        );
        PORT (
            clk      : IN  STD_LOGIC;
            regUpdate: IN  STD_LOGIC;
            regrst   : IN  STD_LOGIC;
            din      : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout     : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    SIGNAL toPop   : STD_LOGIC;
    SIGNAL popRST  : STD_LOGIC;
    SIGNAL regRST  : STD_LOGIC;
    SIGNAL count   : STD_LOGIC_VECTOR(n - 1 DOWNTO 0);

    -- Debug attribute
    ATTRIBUTE MARK_DEBUG : STRING;
    ATTRIBUTE MARK_DEBUG OF Chv_input : SIGNAL IS "TRUE";

BEGIN

    -- XOR input with class vector, gated by run
    toPop   <= (Chv_input XOR hv) AND run;

    -- Reset signals
    popRST  <= rst OR done;
    regRST  <= rst OR run;

    -- Popcount module instantiation
    pop : popCount
        GENERIC MAP (
            lenPop => n
        )
        PORT MAP (
            clk   => clk,
            rst   => popRST,
            en    => toPop,
            dout  => count
        );

    -- Register module to store result
    regPop : reg
        GENERIC MAP (
            lenPop => n
        )
        PORT MAP (
            clk        => clk,
            regUpdate  => done,
            regrst     => regRST,
            din        => count,
            dout       => sim
        );

END ARCHITECTURE behavioral;
