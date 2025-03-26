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

ENTITY encoder IS
    GENERIC (
        d           : INTEGER := 500;     -- Dimension size
        lgf         : INTEGER := 10;      -- Bit width of popCounters output (log2(featureSize))
        featureSize : INTEGER := 700      -- Number of features
    );
    PORT (
        clk         : IN  STD_LOGIC;      -- Clock signal
        rst         : IN  STD_LOGIC;      -- Reset signal
        run         : IN  STD_LOGIC;      -- Run should be '1' for 2 clk cycles
        din         : IN  STD_LOGIC_VECTOR (d - 1 DOWNTO 0); -- Input data
        BV          : IN  STD_LOGIC_VECTOR (d - 1 DOWNTO 0); -- Bit vector for feature selection
        rundegi     : OUT STD_LOGIC;      -- Signal indicating end of round (?)
        done        : OUT STD_LOGIC;      -- Done signal
        ready_M     : OUT STD_LOGIC;      -- Ready signal
        dout        : OUT STD_LOGIC_VECTOR (d - 1 DOWNTO 0) -- Output data
    );
END ENTITY encoder;

ARCHITECTURE behavioral OF encoder IS

    COMPONENT reg IS
        GENERIC (
            lenPop : INTEGER := 8         -- Bit width of reg output
        );
        PORT (
            clk      : IN  STD_LOGIC;
            regUpdate: IN  STD_LOGIC;
            regrst   : IN  STD_LOGIC;
            din      : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
            dout     : OUT STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT XoringInputPop IS
        GENERIC (
            n : INTEGER := 8              -- Number of bits for popcount output
        );
        PORT (
            clk     : IN  STD_LOGIC;
            rst     : IN  STD_LOGIC;
            update  : IN  STD_LOGIC;
            done    : IN  STD_LOGIC;
            din     : IN  STD_LOGIC;
            BV      : IN  STD_LOGIC;
            dout    : OUT STD_LOGIC_VECTOR (n - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT XoringPopCtrl IS
        GENERIC (
            n           : INTEGER := 10;   -- Bit width for popCounter (log2(featureSize))
            featureSize : INTEGER := 700   -- Number of loops
        );
        PORT (
            clk        : IN  STD_LOGIC;
            rst        : IN  STD_LOGIC;
            run        : IN  STD_LOGIC;
            rundegi    : OUT STD_LOGIC;
            update     : OUT STD_LOGIC;
            doneI      : OUT STD_LOGIC;
            doneII     : OUT STD_LOGIC;
            ready_M    : OUT STD_LOGIC
        );
    END COMPONENT;

    SIGNAL update       : STD_LOGIC;
    SIGNAL doneI        : STD_LOGIC;
    CONSTANT test       : STD_LOGIC_VECTOR (lgf - 1 DOWNTO 0) := STD_LOGIC_VECTOR(to_unsigned(featureSize / 2, lgf));
    SIGNAL douttest     : STD_LOGIC_VECTOR (d - 1 DOWNTO 0);
    SIGNAL doutXOR      : STD_LOGIC_VECTOR ((d * lgf) - 1 DOWNTO 0);
    SIGNAL querycheck   : STD_LOGIC_VECTOR (d - 1 DOWNTO 0);
    SIGNAL testID       : STD_LOGIC_VECTOR (999 DOWNTO 0) := (others => '0');
    SIGNAL testBV       : STD_LOGIC_VECTOR (999 DOWNTO 0) := (others => '0');
    SIGNAL IDeq, BVeq, eq : STD_LOGIC;

BEGIN

    -- Instantiate d parallel popcount units
    popCounters : FOR I IN 0 TO d - 1 GENERATE
        pop: XoringInputPop 
            GENERIC MAP (
                n => lgf
            )
            PORT MAP (
                clk     => clk,
                rst     => rst,
                update  => update,
                done    => doneI,
                din     => din(I),
                BV      => BV(I),
                dout    => doutXOR((I + 1) * lgf - 1 DOWNTO I * lgf)
            );
    END GENERATE popCounters;

    -- Instantiate control logic
    ctrl : XoringPopCtrl
        GENERIC MAP (
            n           => lgf,
            featureSize => featureSize
        )
        PORT MAP (
            clk      => clk,
            rst      => rst,
            run      => run,
            rundegi  => rundegi,
            update   => update,
            doneI    => doneI,
            doneII   => done,
            ready_M  => ready_M
        );

    -- Generate output decision: thresholding each lgf-bit counter against "test"
    doutGen : FOR I IN 0 TO d - 1 GENERATE
        dout(I) <= '1' WHEN doutXOR((I + 1) * lgf - 1 DOWNTO I * lgf) > test ELSE '0';
    END GENERATE doutGen;

END ARCHITECTURE behavioral;

